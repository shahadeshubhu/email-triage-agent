# Email Triage Agent

A single Python file (`email_triage_agent.py`) that reads emails, decides what each one needs — reply, task, escalate, or archive — does it, and remembers what it did. Run it on the same inbox as many times as you want: it never repeats itself.

## Run it

```bash
python3 email_triage_agent.py run     # one triage pass
python3 email_triage_agent.py show    # see current state + counts
python3 email_triage_agent.py reset   # wipe state, start clean
```

If your folder is on a filesystem that doesn't support SQLite locking (some mounted/networked drives), redirect state to local disk: `STATE_DIR=/tmp/email_agent python3 email_triage_agent.py run`.

### Real inbox mode

By default it reads 6 fixed fake emails baked into the script, so it runs instantly with no setup. You can also point it at a real, free, disposable inbox:

```bash
TRIAGE_SOURCE=mailtm python3 email_triage_agent.py run
```

First run creates a free address via [mail.tm](https://docs.mail.tm) (no signup needed) and prints it. Send it a real email, run the command again, and it gets fetched, classified, and acted on for real. The address is saved to `mailtm_account.json` so every run checks the same inbox. Running with no new mail correctly does nothing — same guarantee as the stub demo. Confirmed working live via Google Colab.

### Classifier: heuristic (default) or a real LLM

```bash
python3 email_triage_agent.py run                                          # keyword rules, free, offline
TRIAGE_CLASSIFIER=llm ANTHROPIC_API_KEY=sk-... python3 email_triage_agent.py run   # Claude decides
```

If the LLM call fails for any reason (no key, bad key, API error), the agent logs why and falls back to the heuristic instead of crashing — verified directly by running with a missing key and an invalid key, both producing the same output as the pure-heuristic run.

## Proof it's idempotent

Run 1, fresh inbox of 6:

```
Emails fetched:        6
Newly processed:       6
Skipped (dup-guard):   0
  msg-001: replied (reply)
  msg-002: archived (archive)
  msg-003: escalated (escalate)
  msg-004: replied (reply)
  msg-005: task_created (create_task)
  msg-006: archived (archive)
```

Run 2, same inbox, no reset:

```
Emails fetched:        6
Newly processed:       0
Skipped (dup-guard):   6
  msg-001 .. msg-006: skipped_already_handled
```

`show` after both runs: `{'emails': 6, 'tasks': 2, 'replies': 2}` — identical before and after run 2.

A third case simulates a crash: a task gets created but the row never gets marked `handled` (process died mid-action). Re-running finds it unfinished, retries, and a database constraint rejects the duplicate task. Task count stays at 2.

## Design note

**Why SQLite.** The task needs a durable "have I already done this?" check that survives a restart and holds up if two runs happen close together. SQLite gives that for free — one file, no server, and the database itself enforces the rules. A plain text/JSON file can't do this safely: reading, then deciding, then writing back isn't one atomic move, so two runs could both check at the same moment, both see "not done yet," and both act. Keeping state only in memory doesn't survive a restart at all, which fails the "run it twice" requirement outright.

**The trick: claim first, then act — not check, then act.** The naive way is to look up a status, decide, then write. That has a gap: two runs can both look and both see "not done" before either writes anything. Instead, every email's ID is a database primary key, and processing starts by trying to insert that ID. Only one insert for a given ID can ever succeed — the database itself guarantees it. So the "have I seen this?" check and the "claim it" step are literally the same instant, with no gap in between for a second run to sneak through. The same rule is repeated one level down on tasks and replies too, so even if the outer guard were somehow bypassed, a duplicate task or reply still gets rejected by the database itself.

**Edge case — crash mid-action.** If the process dies after claiming an email but before finishing the action, that row is left in a "processing" state. Instead of silently ignoring it forever, the agent retries anything still in that state — safely, because retrying an already-finished action gets rejected by the same claim mechanism above, while retrying an unfinished one completes it. Verified by manually resetting a finished row back to "processing" and confirming a second task doesn't get created.

**Edge case — the source re-sends the same email.** The inbox source can hand back the same message on every check (realistic for a polling API). Dedup is handled entirely on our side, not by trusting the source to only mention new mail once.

**Edge case — retrying with a non-deterministic classifier.** Once an LLM is involved, a retry could theoretically get a different answer than the original classification (an LLM isn't guaranteed to say the same thing twice). If that happened during a crash-retry, an email could end up with two different actions instead of one. Fix: a retry reuses whatever category was already recorded rather than asking the classifier again, so the retry always takes the exact same path as the original attempt.

**How this was tested:**
1. Fresh reset, then run — 6/6 processed correctly.
2. Run again, no reset — 0 processed, all 6 skipped, table counts unchanged.
3. Forced a crash state on one row, re-ran — retried safely, no duplicate task.
4. Every step is logged with a run ID, so any "why didn't this duplicate" question is answerable from the log, not just asserted.
5. Five separate real processes launched at the same instant against the same database, racing over the same 6 emails — every email still only handled once, combined across all five. This is the concurrency proof, covered below in Testing.

## What the actions actually do (and don't)

`action_reply()` and `action_create_task()` write to the database and a log file — they don't send real email or create a real ticket anywhere. Nothing here is wired to SMTP or a task tool. That's a deliberate scope boundary: the assignment is graded on the decide-and-dedup logic, not on delivery, and this keeps the whole thing runnable for free with zero external accounts. mail.tm (the real-inbox mode) is also receive-only, so even there, a "reply" is only ever logged, never actually sent.

There's also no task-assignment model (a task has no owner), and dedup is keyed on message ID, not sender — if the same person emails twice, each message is handled independently in full. Both are reasonable next steps, not oversights.

## Testing

```bash
pip install pytest
pytest -v test_agent.py test_mailtm_source.py
```

Both files sit flat next to `email_triage_agent.py`, no subfolder needed.

`test_agent.py` — six tests, each proving a different way a naive version could break:

1. **Sequential double run** — run, run again, second run touches nothing.
2. **Concurrent race** — 5 real OS processes hit the same database at the same instant; every email still gets handled exactly once combined.
3. **Crash recovery** — forces a mid-crash state, confirms retry doesn't duplicate the task.
4. **Crash recovery + classifier reuse** — confirms a retried email reuses its recorded category instead of reclassifying.
5. **Concurrent resume of a crashed email** — a harder version of #2: 5 threads race to resume the *same* half-finished email at the exact same instant (forced with a threading barrier), and exactly one task gets created.
6. **Growing batch** — feeds new mail alongside already-handled mail between two runs, proving per-email tracking rather than whole-batch detection.

These pass every time, not just usually — the guarantee comes from SQLite's own locking (and an explicit barrier where needed), not from lucky timing.

`test_mailtm_source.py` — three tests on the real-inbox field-mapping logic (mocked HTTP, no live network required). One caught a real bug during development: `subject` and `received_at` were being read from the wrong part of mail.tm's response — fixed.

## Planning loop

For each email: `fetch → claim → classify → decide → act → record`. Every step is logged with a run ID.

Classification defaults to keyword rules so everything runs offline for free, but can dispatch to a real Claude call instead (see "Classifier" above).

## Files

- `email_triage_agent.py` — the agent
- `test_agent.py`, `test_mailtm_source.py` — the test suite
- `demo_colab.ipynb` — a runnable walkthrough (Google Colab) covering the basic demo, the test suite, and a section for sending real test emails
- `README.md` — this file

Produced by a run: `state.db` (source of truth), `sent_replies.log` / `tasks_created.log` (human-readable logs), `mailtm_account.json` (real-inbox credentials, only in mailtm mode).

## What I'd do next with more time

- **Make actions real** — wire replies to a real email API, tasks to a real task tool. Both are single-function swaps behind the existing dispatch table; the idempotency guarantees wouldn't need to change.
- **Task ownership** — add routing logic and an assignee field.
- **Harden the LLM classifier** — move to structured output instead of parsed text, add few-shot examples, and guard against prompt injection from email content.
- **Observability** — the log already captures every decision; a small dashboard would make it viewable without a database client.
- **Scale past SQLite** — the right choice at this size, but Postgres with the same claim-first pattern would carry the same guarantee at higher volume.
