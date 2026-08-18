"""
Automated proof of idempotency for the email triage agent.

These tests don't just re-run the agent twice in sequence and eyeball the
output — the sequential case is table stakes. Two of them push harder:
`test_concurrent_race_no_duplicates` launches several real, separate OS
processes against the SAME database at the SAME time (a cron overlap or
retry storm), and `test_concurrent_resume_of_crashed_email_no_duplicate`
does the same but for the *resume* code path specifically, which is
guarded by a different mechanism (an action-level UNIQUE constraint) than
the initial claim (a PRIMARY KEY insert). A third,
`test_growing_batch_only_new_emails_processed`, checks per-email state
tracking rather than whole-batch detection, by feeding the agent a batch
that grows between two runs instead of replaying an identical one.

Run with:
    pip install pytest
    pytest -v test_agent.py
"""

import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

SCRIPT = str(Path(__file__).resolve().parent / "email_triage_agent.py")
TOTAL_EMAILS = 6          # size of the stub inbox in email_triage_agent.py
EXPECTED_TASKS = 2        # msg-005 (create_task) + msg-003 (escalate)
EXPECTED_REPLIES = 2      # msg-001, msg-004


def run_agent(state_dir, cmd="run", timeout=30):
    env = os.environ.copy()
    env["STATE_DIR"] = str(state_dir)
    return subprocess.run(
        [sys.executable, SCRIPT, cmd],
        capture_output=True, text=True, env=env, timeout=timeout,
    )


def db_counts(state_dir):
    conn = sqlite3.connect(Path(state_dir) / "state.db")
    counts = {
        "emails": conn.execute("SELECT COUNT(*) FROM emails").fetchone()[0],
        "handled": conn.execute("SELECT COUNT(*) FROM emails WHERE status='handled'").fetchone()[0],
        "tasks": conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0],
        "replies": conn.execute("SELECT COUNT(*) FROM replies").fetchone()[0],
    }
    conn.close()
    return counts


def newly_processed_count(stdout):
    m = re.search(r"Newly processed:\s+(\d+)", stdout)
    assert m, f"couldn't parse summary from:\n{stdout}"
    return int(m.group(1))


# --------------------------------------------------------------------------
# 1. Sequential re-run: the baseline case the task asks for explicitly.
# --------------------------------------------------------------------------

def test_sequential_double_run_no_duplicates(tmp_path):
    first = run_agent(tmp_path, "run")
    assert newly_processed_count(first.stdout) == TOTAL_EMAILS

    second = run_agent(tmp_path, "run")
    assert newly_processed_count(second.stdout) == 0
    assert f"Skipped (dup-guard):   {TOTAL_EMAILS}" in second.stdout

    counts = db_counts(tmp_path)
    assert counts == {
        "emails": TOTAL_EMAILS,
        "handled": TOTAL_EMAILS,
        "tasks": EXPECTED_TASKS,
        "replies": EXPECTED_REPLIES,
    }


# --------------------------------------------------------------------------
# 2. True concurrency: N separate OS processes fired at the same instant
#    against the same database. This is the test that actually stresses the
#    claim-before-act / UNIQUE-constraint design, not just the "run it
#    twice by hand" happy path.
# --------------------------------------------------------------------------

def test_concurrent_race_no_duplicates(tmp_path):
    N = 5
    with ThreadPoolExecutor(max_workers=N) as pool:
        results = list(pool.map(lambda _: run_agent(tmp_path, "run"), range(N)))

    for r in results:
        assert r.returncode == 0, r.stderr

    # Across ALL N racing processes combined, every email must have been
    # newly processed exactly once in total -- not zero times (lost) and
    # not more than once (duplicated).
    total_newly_processed = sum(newly_processed_count(r.stdout) for r in results)
    assert total_newly_processed == TOTAL_EMAILS, (
        f"expected exactly {TOTAL_EMAILS} emails processed across all "
        f"{N} racing processes combined, got {total_newly_processed} "
        "-- this would mean either lost work or duplicate side effects"
    )

    counts = db_counts(tmp_path)
    assert counts == {
        "emails": TOTAL_EMAILS,
        "handled": TOTAL_EMAILS,
        "tasks": EXPECTED_TASKS,
        "replies": EXPECTED_REPLIES,
    }, f"race produced duplicate/missing side effects: {counts}"


# --------------------------------------------------------------------------
# 3. Crash recovery: a run dies after the action succeeded but before the
#    row was marked 'handled'. The next run must retry safely, not skip the
#    email forever and not duplicate the already-created task.
# --------------------------------------------------------------------------

def test_crash_recovery_no_duplicate_task(tmp_path):
    run_agent(tmp_path, "run")
    before = db_counts(tmp_path)

    # Simulate a crash: task already exists, but status was never flipped
    # to 'handled' because the process died right before that final write.
    conn = sqlite3.connect(Path(tmp_path) / "state.db")
    conn.execute("UPDATE emails SET status='processing', handled_at=NULL WHERE id='msg-005'")
    conn.commit()
    conn.close()

    recovery = run_agent(tmp_path, "run")
    assert "already_task" in recovery.stdout

    after = db_counts(tmp_path)
    assert after["tasks"] == before["tasks"], "crash recovery must not duplicate the task"
    assert after["handled"] == TOTAL_EMAILS, "recovered email must end up handled"


# --------------------------------------------------------------------------
# 4. Category reuse on retry: the classifier (especially the optional LLM
#    backend) is not guaranteed to be perfectly deterministic. If a retry
#    reclassified from scratch, a non-deterministic answer could send the
#    same email down a *different* action path than the original attempt
#    (e.g. task already exists, retry reclassifies as "reply" and now sends
#    one too). The agent must reuse the category it already recorded for a
#    resumed email instead of reclassifying it.
# --------------------------------------------------------------------------

def test_crash_recovery_reuses_recorded_category_no_reclassify(tmp_path):
    run_agent(tmp_path, "run")

    conn = sqlite3.connect(Path(tmp_path) / "state.db")
    row = conn.execute("SELECT category FROM emails WHERE id='msg-005'").fetchone()
    original_category = row[0]
    assert original_category == "create_task"

    # Simulate a crash after classification+action succeeded but before the
    # row was marked handled -- category is already on record.
    conn.execute("UPDATE emails SET status='processing', handled_at=NULL WHERE id='msg-005'")
    conn.commit()
    conn.close()

    recovery = run_agent(tmp_path, "run")

    conn = sqlite3.connect(Path(tmp_path) / "state.db")
    log_rows = conn.execute(
        "SELECT step, detail FROM run_log WHERE email_id='msg-005' ORDER BY id"
    ).fetchall()
    conn.close()

    steps = [s for s, _ in log_rows]
    assert "reusing_recorded_category" in steps, (
        f"retry should reuse the recorded category instead of reclassifying, got steps: {steps}"
    )
    # It must NOT have run a fresh 'classified' step during the retry pass
    # (only the very first, pre-crash run should have logged one).
    classify_events = [d for s, d in log_rows if s == "classified"]
    assert classify_events.count(original_category) == 1, (
        "email should only be classified once total, not reclassified on retry"
    )


# --------------------------------------------------------------------------
# 5. Concurrent RESUME race: harder than test 2 above. Test 2 races several
#    processes over a brand-new email, which is guarded by the atomic
#    PRIMARY KEY INSERT in the "claim" step. But process_email's resume
#    branch (for a row already sitting in status='processing') does NOT
#    attempt another atomic INSERT -- it just reads the status and falls
#    through to classify+act. If two callers reach that branch for the
#    SAME row at the same time, they could both fall through concurrently.
#    This test forces exactly that: the email's category is already
#    recorded (crash happened after classification) but its task was
#    never actually created (crash happened before the action), so N
#    threads all race to be the one that actually creates it.
#
#    This uses real threads + a synchronization barrier rather than
#    separate subprocesses (contrast with test 2). An earlier subprocess-
#    based version of this test was flaky in exactly the interesting way:
#    subprocess/interpreter startup overhead (tens of ms) dwarfs the
#    actual DB critical section (microseconds), so one process usually
#    finished the whole resume before the next one even checked the
#    status -- the race window was real but too narrow to hit reliably by
#    chance. A barrier forces all N callers to hit process_email() at the
#    exact same instant, making the race deterministic instead of
#    timing-dependent. Each thread opens its own DB connection, matching
#    how the real CLI does it once per run() call; the action-level
#    UNIQUE constraint on tasks.email_id is what has to hold here, not
#    the emails PRIMARY KEY (which was already satisfied before this test
#    even starts).
# --------------------------------------------------------------------------

def test_concurrent_resume_of_crashed_email_no_duplicate(tmp_path, monkeypatch):
    import threading
    import email_triage_agent as agent

    monkeypatch.setattr(agent, "DB_PATH", tmp_path / "state.db")
    monkeypatch.setattr(agent, "REPLIES_LOG", tmp_path / "sent_replies.log")
    monkeypatch.setattr(agent, "TASKS_LOG", tmp_path / "tasks_created.log")
    agent.init_db()

    email = {
        "id": "crash-001", "sender": "x@example.test", "subject": "please help",
        "body": "please help me set this up", "received_at": "2026-08-18T00:00:00Z",
    }
    # Seed a row already classified but not yet acted on -- category is
    # recorded, no task exists yet, status is still 'processing'.
    conn = agent.get_conn()
    conn.execute(
        "INSERT INTO emails (id, sender, subject, body, received_at, status, claimed_at, category) "
        "VALUES (?,?,?,?,?, 'processing', ?, 'create_task')",
        (email["id"], email["sender"], email["subject"], email["body"], email["received_at"], agent.now_iso()),
    )
    conn.commit()
    conn.close()

    N = 5
    barrier = threading.Barrier(N)
    results = [None] * N

    def worker(i):
        conn = agent.get_conn()
        barrier.wait()  # force all N threads into process_email() at the same instant
        results[i] = agent.process_email(conn, f"resume{i}", email)
        conn.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    conn = agent.get_conn()
    task_count = conn.execute("SELECT COUNT(*) FROM tasks WHERE email_id=?", (email["id"],)).fetchone()[0]
    conn.close()
    assert task_count == 1, f"exactly one task should exist after {N} racing resumers, got {task_count}"

    # Exactly one thread should have actually created the task. The other
    # N-1 can land on either "already_task" (caught the IntegrityError on
    # its own insert attempt) or "skipped_already_handled" (the winner had
    # already committed status='handled' by the time this thread's SELECT
    # ran) -- both are safe, non-duplicating outcomes. Which one occurs is
    # a Python GIL/scheduling detail, not something to hard-code: in
    # practice the GIL tends to let the winning thread run to completion
    # before the others get scheduled, so "skipped_already_handled" is
    # actually the more common outcome here despite the barrier forcing
    # simultaneous entry -- the barrier proves the entry race is real, it
    # doesn't guarantee every loser is caught mid-flight.
    outcomes = [r["outcome"] for r in results]
    assert outcomes.count("task_created") == 1, f"expected exactly 1 winner, got outcomes={outcomes}"
    non_winners = [o for o in outcomes if o != "task_created"]
    assert len(non_winners) == N - 1
    assert all(o in ("already_task", "skipped_already_handled") for o in non_winners), (
        f"every non-winning thread must land on a safe no-op, got outcomes={outcomes}"
    )


# --------------------------------------------------------------------------
# 6. Growing batch: a more realistic scenario than replaying an identical
#    batch twice (test 1 above). Real inboxes grow between polls -- new
#    mail arrives alongside emails already handled, exactly what happened
#    when testing this live against a real mail.tm inbox. This proves
#    per-email state tracking, not just whole-batch detection: an
#    implementation that (incorrectly) fingerprinted "have I seen this
#    exact batch of IDs before" rather than tracking each email's own
#    state independently would pass test 1 but fail this one.
#
#    This test calls process_email() directly rather than going through
#    the CLI/subprocess, since the built-in stub inbox is a fixed list --
#    calling the function directly is the simplest way to feed it a
#    custom, growing batch without adding test-only hooks to the
#    production script.
# --------------------------------------------------------------------------

def test_growing_batch_only_new_emails_processed(tmp_path, monkeypatch):
    import email_triage_agent as agent

    monkeypatch.setattr(agent, "DB_PATH", tmp_path / "state.db")
    monkeypatch.setattr(agent, "REPLIES_LOG", tmp_path / "sent_replies.log")
    monkeypatch.setattr(agent, "TASKS_LOG", tmp_path / "tasks_created.log")
    agent.init_db()

    def make_email(id_, subject, body):
        return {
            "id": id_, "sender": "x@example.test", "subject": subject,
            "body": body, "received_at": "2026-08-18T00:00:00Z",
        }

    batch_1 = [
        make_email("g-001", "Question?", "what are your hours?"),
        make_email("g-002", "Question?", "how do I reset my password?"),
    ]

    conn = agent.get_conn()
    results_1 = [agent.process_email(conn, "run1", e) for e in batch_1]
    conn.close()
    assert all(r["outcome"] != "skipped_already_handled" for r in results_1)

    # Batch grows: same two emails plus one genuinely new one, mirroring a
    # real inbox re-polled after new mail arrived.
    batch_2 = batch_1 + [make_email("g-003", "urgent outage", "production is down")]

    conn = agent.get_conn()
    results_2 = [agent.process_email(conn, "run2", e) for e in batch_2]
    conn.close()

    outcomes_2 = {r["id"]: r["outcome"] for r in results_2}
    assert outcomes_2["g-001"] == "skipped_already_handled"
    assert outcomes_2["g-002"] == "skipped_already_handled"
    assert outcomes_2["g-003"] != "skipped_already_handled"

    conn = agent.get_conn()
    total_handled = conn.execute("SELECT COUNT(*) FROM emails WHERE status='handled'").fetchone()[0]
    conn.close()
    assert total_handled == 3, "old emails must stay handled, new one must be added -- no re-processing, no loss"
