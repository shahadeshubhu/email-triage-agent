#!/usr/bin/env python3
"""
Email Triage Agent — reads emails, decides on an action, and NEVER repeats
itself across runs.

Usage:
    python email_triage_agent.py run              # one triage pass
    python email_triage_agent.py show              # dump current state
    python email_triage_agent.py reset             # wipe state (fresh demo)

Classifier backend (env vars):
    TRIAGE_CLASSIFIER=heuristic   (default) keyword rules, offline, free
    TRIAGE_CLASSIFIER=llm         calls Claude via ANTHROPIC_API_KEY; falls
                                   back to the heuristic on any failure

Email source (env vars):
    TRIAGE_SOURCE=stub            (default) 6 built-in fake emails, offline
    TRIAGE_SOURCE=mailtm          real inbox via the free mail.tm API (no
                                   signup, no key). First run prints a real
                                   email address -- send test mail to it,
                                   then run again to see it get triaged.

------------------------------------------------------------------------------
WHY SQLITE + HOW IDEMPOTENCY WORKS (full reasoning + how it was tested is in
README.md -- this is just the mechanism, short version)

State lives in one SQLite file. It gives atomic, durable "have I done this
already?" checks for free -- no server, no hand-rolled locking.

The trick is claim-before-act, not check-then-act:
  1. `emails.id` is a PRIMARY KEY. Processing starts by INSERTing that id.
     Success = genuinely new. Failure (IntegrityError) = already claimed,
     skip immediately. The check and the claim are the same operation, so
     there's no gap where two runs could both think an email is new.
  2. `tasks.email_id` / `replies.email_id` are also UNIQUE, as a second
     guard even if step 1 were ever bypassed.
  3. Status flips to 'handled' only after the action succeeds. A crash
     mid-action leaves it at 'processing'; the next run finds that and
     retries -- safely, since the UNIQUE guards make a retried action a
     no-op if it already went through.

Result: run this as many times as you want on the same inbox -- every run
after the first creates zero new tasks, replies, or duplicate records.
------------------------------------------------------------------------------
"""

import sqlite3
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# State is written next to this script by default. Some sandboxed/networked
# filesystems (e.g. certain mounted cloud-sync folders) don't support
# SQLite's file locking; if you hit "disk I/O error", point STATE_DIR at a
# local/native disk path instead, e.g.:
#   STATE_DIR=/tmp/email_agent python3 email_triage_agent.py run
import os
BASE_DIR = Path(os.environ.get("STATE_DIR", str(Path(__file__).resolve().parent)))
BASE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = BASE_DIR / "state.db"
REPLIES_LOG = BASE_DIR / "sent_replies.log"
TASKS_LOG = BASE_DIR / "tasks_created.log"
MAILTM_ACCOUNT_FILE = BASE_DIR / "mailtm_account.json"


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Storage layer
# --------------------------------------------------------------------------

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS emails (
            id            TEXT PRIMARY KEY,
            sender        TEXT NOT NULL,
            subject       TEXT NOT NULL,
            body          TEXT NOT NULL,
            received_at   TEXT NOT NULL,
            category      TEXT,
            action        TEXT,
            status        TEXT NOT NULL DEFAULT 'processing',
            claimed_at    TEXT NOT NULL,
            handled_at    TEXT
        );

        CREATE TABLE IF NOT EXISTS tasks (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            email_id      TEXT NOT NULL UNIQUE REFERENCES emails(id),
            title         TEXT NOT NULL,
            priority      TEXT NOT NULL,
            created_at    TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS replies (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            email_id      TEXT NOT NULL UNIQUE REFERENCES emails(id),
            reply_text    TEXT NOT NULL,
            sent_at       TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS run_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id        TEXT NOT NULL,
            email_id      TEXT,
            step          TEXT NOT NULL,
            detail        TEXT,
            ts            TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()


def log_step(conn, run_id, email_id, step, detail=""):
    conn.execute(
        "INSERT INTO run_log (run_id, email_id, step, detail, ts) VALUES (?,?,?,?,?)",
        (run_id, email_id, step, detail, now_iso()),
    )


# --------------------------------------------------------------------------
# Email source -- two backends behind one function:
#   stub    (default) 6 fixed fake emails, fully offline.
#   mailtm  a real free inbox via mail.tm (https://docs.mail.tm), no signup.
#           Creates one disposable address on first run, saves the
#           credentials locally so later runs poll the same inbox. mail.tm
#           has no "mark as fetched" concept, so it re-returns the same
#           messages every time -- realistic polling behavior, and exactly
#           the case our own dedup state (not the source) is meant to handle.
# --------------------------------------------------------------------------

MAILTM_API = "https://api.mail.tm"


def _mailtm_request(method, path, payload=None, token=None, timeout=15):
    import urllib.request
    import urllib.error

    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"{MAILTM_API}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"mail.tm {method} {path} failed: HTTP {e.code} {e.read().decode()[:200]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"mail.tm {method} {path} unreachable: {e.reason}") from e


def get_or_create_mailtm_account():
    """Reuses the same disposable inbox across runs (saved to
    mailtm_account.json) so a test email sent after run N is still there
    to be found on run N+1. `reset` deletes this file along with the rest
    of the agent's state, so a fresh run after reset gets a brand new
    address."""
    if MAILTM_ACCOUNT_FILE.exists():
        return json.loads(MAILTM_ACCOUNT_FILE.read_text())

    domains = _mailtm_request("GET", "/domains")
    members = domains.get("hydra:member") or []
    if not members:
        raise RuntimeError("mail.tm returned no available domains")
    domain = members[0]["domain"]

    address = f"triage-agent-{uuid.uuid4().hex[:10]}@{domain}"
    password = uuid.uuid4().hex
    _mailtm_request("POST", "/accounts", {"address": address, "password": password})

    account = {"address": address, "password": password}
    MAILTM_ACCOUNT_FILE.write_text(json.dumps(account))
    return account


def _strip_html(html_field):
    import re
    text = " ".join(html_field) if isinstance(html_field, list) else str(html_field or "")
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


def fetch_new_emails_mailtm():
    account = get_or_create_mailtm_account()
    token = _mailtm_request("POST", "/token", {"address": account["address"], "password": account["password"]})["token"]

    listing = _mailtm_request("GET", "/messages", token=token)
    members = listing.get("hydra:member") or []

    emails = []
    for m in members:
        # The list endpoint only gives a truncated preview ("intro"); fetch
        # the full message for the real body.
        detail = _mailtm_request("GET", f"/messages/{m['id']}", token=token)
        # Detail is authoritative; list entry is only a fallback in case a
        # field is ever missing from the detail response.
        body = detail.get("text") or _strip_html(detail.get("html")) or detail.get("intro") or ""
        sender = ((detail.get("from") or m.get("from")) or {}).get("address", "unknown@unknown")
        emails.append({
            "id": m["id"],  # mail.tm's own message id -- stable, unique, exactly what our claim-before-act guard needs
            "sender": sender,
            "subject": detail.get("subject") or m.get("subject") or "(no subject)",
            "body": body,
            "received_at": detail.get("createdAt") or m.get("createdAt") or now_iso(),
        })
    return emails, account["address"]


def fetch_new_emails_stub():
    return [
        {
            "id": "msg-001",
            "sender": "alice@customer.com",
            "subject": "Can you send me the Q3 invoice?",
            "body": "Hi, could you please send over the Q3 invoice when you get a chance? Thanks!",
            "received_at": "2026-08-14T09:12:00Z",
        },
        {
            "id": "msg-002",
            "sender": "no-reply@newsletter.io",
            "subject": "Your weekly digest is here",
            "body": "Check out this week's top stories and updates from our platform.",
            "received_at": "2026-08-14T10:03:00Z",
        },
        {
            "id": "msg-003",
            "sender": "ops@vendor-systems.com",
            "subject": "URGENT: production outage affecting billing",
            "body": "We are seeing a critical outage impacting billing for all customers. Need immediate attention.",
            "received_at": "2026-08-14T11:47:00Z",
        },
        {
            "id": "msg-004",
            "sender": "bob@partner.org",
            "subject": "Question about API rate limits",
            "body": "Quick question - what are the current rate limits on the v2 API? Trying to plan our integration.",
            "received_at": "2026-08-14T13:20:00Z",
        },
        {
            "id": "msg-005",
            "sender": "hr@mycompany.com",
            "subject": "Please review and sign the updated policy doc",
            "body": "Please review the attached policy update and sign by end of week.",
            "received_at": "2026-08-14T14:05:00Z",
        },
        {
            "id": "msg-006",
            "sender": "deals@shopping-promo.com",
            "subject": "50% off everything this weekend only!!!",
            "body": "Don't miss our biggest sale of the year. Shop now and save big.",
            "received_at": "2026-08-14T15:00:00Z",
        },
    ]


def fetch_new_emails():
    """Dispatches to the configured source backend. Returns just the list
    of emails (any real-inbox address to display is printed here so `run()`
    doesn't need to know which backend is active)."""
    source = os.environ.get("TRIAGE_SOURCE", "stub").lower()
    if source == "mailtm":
        emails, address = fetch_new_emails_mailtm()
        print(f"[mail.tm] polling inbox: {address}")
        print(f"[mail.tm] send a test email to this address, then run again to see it triaged")
        return emails
    return fetch_new_emails_stub()


# --------------------------------------------------------------------------
# Classification -- two backends behind one function:
#   heuristic  (default) keyword rules, deterministic, free, offline.
#   llm        asks Claude to judge each email. Opt in with
#              TRIAGE_CLASSIFIER=llm + ANTHROPIC_API_KEY. Any failure
#              (missing key, bad key, API error) falls back to the
#              heuristic and logs why, instead of crashing the run.
#
# VALID_CATEGORIES is the fixed vocabulary both backends must return.
# --------------------------------------------------------------------------

VALID_CATEGORIES = {"reply", "create_task", "archive", "escalate"}

URGENT_WORDS = {"urgent", "outage", "critical", "immediately", "asap", "down"}
PROMO_SENDER_HINTS = {"no-reply", "newsletter", "promo", "deals", "marketing"}
REQUEST_WORDS = {"please", "could you", "can you", "need you to", "review and sign"}
QUESTION_HINTS = {"?", "question about", "what are", "how do", "when will"}


def classify_email_heuristic(email):
    subject = email["subject"].lower()
    body = email["body"].lower()
    sender = email["sender"].lower()
    text = subject + " " + body

    if any(w in text for w in URGENT_WORDS):
        return "escalate"
    if any(hint in sender for hint in PROMO_SENDER_HINTS):
        return "archive"
    if any(q in text for q in QUESTION_HINTS):
        return "reply"
    if any(r in text for r in REQUEST_WORDS):
        return "create_task"
    return "archive"


_LLM_SYSTEM_PROMPT = (
    "You triage inbound email. Read the message and decide exactly one "
    "action from this fixed set: reply, create_task, archive, escalate.\n"
    "- reply: sender asked a direct question you can acknowledge\n"
    "- create_task: sender is requesting something that needs follow-up work\n"
    "- escalate: urgent, time-critical, or an outage/incident\n"
    "- archive: no action needed (newsletters, promos, FYI-only)\n"
    "Respond with ONLY the single category word, nothing else."
)


def classify_email_llm(email):
    """Ask Claude to classify the email. Raises on any failure so the
    caller can decide how to fall back — this function never silently
    guesses."""
    import anthropic  # imported lazily so the heuristic path has zero dependency on it

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        temperature=0,
        system=_LLM_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"From: {email['sender']}\n"
                f"Subject: {email['subject']}\n"
                f"Body: {email['body']}"
            ),
        }],
    )
    category = message.content[0].text.strip().lower()
    if category not in VALID_CATEGORIES:
        raise ValueError(f"model returned an unrecognized category: {category!r}")
    return category


def classify_email(email, conn=None, run_id=None, email_id=None):
    """Dispatches to the configured classifier backend, falling back to
    the heuristic (and logging why) if the LLM backend fails."""
    backend = os.environ.get("TRIAGE_CLASSIFIER", "heuristic").lower()
    if backend == "llm":
        try:
            return classify_email_llm(email)
        except Exception as e:
            if conn is not None:
                log_step(conn, run_id, email_id, "classifier_fallback",
                          f"llm classifier failed ({e!r}), using heuristic")
            return classify_email_heuristic(email)
    return classify_email_heuristic(email)


# --------------------------------------------------------------------------
# Actions — each is guarded by a UNIQUE(email_id) constraint so retrying
# an action for an email that already has a task/reply is a silent no-op,
# not a duplicate.
# --------------------------------------------------------------------------

def action_reply(conn, run_id, email):
    try:
        reply_text = (
            f"Hi, thanks for your note about \"{email['subject']}\" — "
            "someone from our team will follow up shortly with details."
        )
        conn.execute(
            "INSERT INTO replies (email_id, reply_text, sent_at) VALUES (?,?,?)",
            (email["id"], reply_text, now_iso()),
        )
        with open(REPLIES_LOG, "a") as f:
            f.write(f"[{now_iso()}] To: {email['sender']} | Re: {email['subject']}\n{reply_text}\n\n")
        log_step(conn, run_id, email["id"], "action_executed", f"reply sent: {reply_text[:60]}...")
        return "replied"
    except sqlite3.IntegrityError:
        log_step(conn, run_id, email["id"], "action_skipped_duplicate", "reply already exists")
        return "already_replied"


def action_create_task(conn, run_id, email, priority="normal"):
    try:
        title = f"Follow up: {email['subject']}"
        conn.execute(
            "INSERT INTO tasks (email_id, title, priority, created_at) VALUES (?,?,?,?)",
            (email["id"], title, priority, now_iso()),
        )
        with open(TASKS_LOG, "a") as f:
            f.write(f"[{now_iso()}] priority={priority} | {title} (from {email['sender']})\n")
        log_step(conn, run_id, email["id"], "action_executed", f"task created: {title}")
        return "task_created"
    except sqlite3.IntegrityError:
        log_step(conn, run_id, email["id"], "action_skipped_duplicate", "task already exists")
        return "already_task"


def action_archive(conn, run_id, email):
    log_step(conn, run_id, email["id"], "action_executed", "archived, no external side effect")
    return "archived"


def action_escalate(conn, run_id, email):
    result = action_create_task(conn, run_id, email, priority="high")
    log_step(conn, run_id, email["id"], "action_executed", "escalated as high-priority task")
    return "escalated" if result == "task_created" else "already_escalated"


ACTION_MAP = {
    "reply": action_reply,
    "create_task": action_create_task,
    "archive": action_archive,
    "escalate": action_escalate,
}


# --------------------------------------------------------------------------
# Planner — this is the multi-step decision loop:
#   fetch -> claim (idempotency gate) -> classify -> decide -> execute -> record
# --------------------------------------------------------------------------

def process_email(conn, run_id, email):
    # Step 1: attempt to atomically claim this email id. This is the single
    # idempotency gate for "have we ever started handling this email".
    resuming = False
    try:
        conn.execute(
            "INSERT INTO emails (id, sender, subject, body, received_at, status, claimed_at) "
            "VALUES (?,?,?,?,?, 'processing', ?)",
            (email["id"], email["sender"], email["subject"], email["body"],
             email["received_at"], now_iso()),
        )
        conn.commit()
        log_step(conn, run_id, email["id"], "claimed", "new email, claimed for processing")
    except sqlite3.IntegrityError:
        # Already claimed by this or a prior run. Check whether it was fully
        # handled or left mid-flight (crash recovery case).
        row = conn.execute("SELECT status FROM emails WHERE id=?", (email["id"],)).fetchone()
        if row and row["status"] == "handled":
            log_step(conn, run_id, email["id"], "skipped_duplicate", "already handled, no action taken")
            conn.commit()
            return {"id": email["id"], "outcome": "skipped_already_handled"}
        else:
            resuming = True
            log_step(conn, run_id, email["id"], "resuming_incomplete", "found in 'processing' state, retrying action")
            conn.commit()
            # fall through to act again; action inserts are UNIQUE-guarded
            # so this cannot double-create anything.

    # Step 2: classify. If we're resuming a crashed run that already has a
    # recorded category, REUSE it instead of asking the classifier again --
    # an LLM isn't guaranteed to answer the same way twice, and a fresh
    # answer could give this email two different actions (task from
    # attempt 1, reply from attempt 2). Reusing keeps a retry identical to
    # the original decision.
    existing = conn.execute("SELECT category FROM emails WHERE id=?", (email["id"],)).fetchone()
    if resuming and existing and existing["category"]:
        category = existing["category"]
        log_step(conn, run_id, email["id"], "reusing_recorded_category",
                  f"skipping reclassification, retry uses prior decision: {category}")
    else:
        category = classify_email(email, conn=conn, run_id=run_id, email_id=email["id"])
        conn.execute("UPDATE emails SET category=? WHERE id=?", (category, email["id"]))
        log_step(conn, run_id, email["id"], "classified", category)
        conn.commit()

    # Step 3: decide + execute
    handler = ACTION_MAP[category]
    outcome = handler(conn, run_id, email)

    # Step 4: record completion
    conn.execute(
        "UPDATE emails SET action=?, status='handled', handled_at=? WHERE id=?",
        (category, now_iso(), email["id"]),
    )
    log_step(conn, run_id, email["id"], "recorded_handled", outcome)
    conn.commit()

    return {"id": email["id"], "category": category, "outcome": outcome}


def run():
    init_db()
    run_id = str(uuid.uuid4())[:8]
    conn = get_conn()
    try:
        emails = fetch_new_emails()
    except Exception as e:
        # A source outage (e.g. no network) should not touch any state --
        # nothing was claimed, nothing was acted on, so it's safe to just
        # report the failure and stop before writing anything.
        print(f"Failed to fetch emails: {e}")
        conn.close()
        return []

    log_step(conn, run_id, None, "run_started", f"{len(emails)} emails fetched from source")
    conn.commit()

    results = [process_email(conn, run_id, e) for e in emails]

    log_step(conn, run_id, None, "run_finished", json.dumps(results))
    conn.commit()
    conn.close()

    new = [r for r in results if r["outcome"] not in ("skipped_already_handled",)]
    skipped = [r for r in results if r["outcome"] == "skipped_already_handled"]
    classifier_backend = os.environ.get("TRIAGE_CLASSIFIER", "heuristic").lower()
    source_backend = os.environ.get("TRIAGE_SOURCE", "stub").lower()
    print(f"\n=== Run {run_id} summary ===")
    print(f"  Timestamp:             {now_iso()}")
    print(f"  Source:                {source_backend}")
    print(f"  Classifier:            {classifier_backend}")
    print(f"  Emails fetched:        {len(emails)}")
    print(f"  Newly processed:       {len(new)}")
    print(f"  Skipped (dup-guard):   {len(skipped)}")
    for r in results:
        print(f"    - {r['id']}: {r['outcome']}" + (f" ({r.get('category')})" if r.get('category') else ""))
    return results


def show():
    init_db()
    conn = get_conn()
    print("\n-- emails --")
    for row in conn.execute(
        "SELECT id, sender, subject, category, action, status FROM emails ORDER BY claimed_at"
    ):
        print(dict(row))
    print("\n-- tasks --")
    for row in conn.execute("SELECT email_id, title, priority FROM tasks ORDER BY id"):
        print(dict(row))
    print("\n-- replies --")
    for row in conn.execute("SELECT email_id, sent_at FROM replies ORDER BY id"):
        print(dict(row))
    counts = {
        "emails": conn.execute("SELECT COUNT(*) c FROM emails").fetchone()["c"],
        "tasks": conn.execute("SELECT COUNT(*) c FROM tasks").fetchone()["c"],
        "replies": conn.execute("SELECT COUNT(*) c FROM replies").fetchone()["c"],
    }
    print(f"\nCounts: {counts}")
    conn.close()


def reset():
    for p in (DB_PATH, REPLIES_LOG, TASKS_LOG, MAILTM_ACCOUNT_FILE):
        if p.exists():
            p.unlink()
    print("State wiped.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        run()
    elif cmd == "show":
        show()
    elif cmd == "reset":
        reset()
    else:
        print("Usage: email_triage_agent.py [run|show|reset]")
