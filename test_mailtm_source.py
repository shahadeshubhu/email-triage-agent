"""
Offline verification of the mail.tm real-inbox integration.

This does NOT hit the network. It mocks email_triage_agent._mailtm_request
with response shapes taken directly from mail.tm's published API schema
(https://api.mail.tm/docs.jsonld -- the #Message class documents msgid,
from, to, subject, intro, createdAt; the detail endpoint additionally
returns text/html for the full body per mail.tm's documented behavior).

Why mocked instead of live: this was built in a sandboxed environment
whose outbound network is allowlisted and does not include api.mail.tm, so
the real HTTP round-trip could not be exercised from there. This test
proves the *parsing/mapping logic* is correct against the documented
schema; the actual live round-trip should be confirmed once by running
`TRIAGE_SOURCE=mailtm python3 email_triage_agent.py run` on a machine with
normal internet access -- it will print a real, working email address.
"""

import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import email_triage_agent as agent


def test_fetch_new_emails_mailtm_maps_fields_correctly(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "MAILTM_ACCOUNT_FILE", tmp_path / "mailtm_account.json")

    calls = []

    def fake_request(method, path, payload=None, token=None, timeout=15):
        calls.append((method, path))
        if path == "/domains":
            return {"hydra:member": [{"domain": "example.test"}]}
        if path == "/accounts":
            return {"address": payload["address"]}
        if path == "/token":
            assert token is None
            return {"token": "fake-jwt-token"}
        if path == "/messages":
            assert token == "fake-jwt-token"
            return {
                "hydra:member": [
                    {"id": "abc123", "from": {"address": "list-preview@should-be-overridden.test"}, "subject": "Hi", "createdAt": "2026-08-16T10:00:00+00:00"},
                ]
            }
        if path == "/messages/abc123":
            return {
                "id": "abc123",
                "from": {"address": "real.sender@example.test"},
                "subject": "Can you help with something?",
                "text": "Full plain-text body of the message.",
                "html": ["<p>Full <b>html</b> body</p>"],
                "intro": "Full plain-text bo...",
                "createdAt": "2026-08-16T10:00:00+00:00",
            }
        raise AssertionError(f"unexpected mail.tm call: {method} {path}")

    monkeypatch.setattr(agent, "_mailtm_request", fake_request)

    emails, address = agent.fetch_new_emails_mailtm()

    assert address.endswith("@example.test")
    assert len(emails) == 1
    e = emails[0]
    # id must be mail.tm's own message id -- this is what our idempotency
    # PRIMARY KEY relies on to be stable and unique.
    assert e["id"] == "abc123"
    # sender must come from the message DETAIL, not the truncated list
    # preview, since the detail is authoritative.
    assert e["sender"] == "real.sender@example.test"
    assert e["subject"] == "Can you help with something?"
    # prefers "text" over "html"/"intro" when present
    assert e["body"] == "Full plain-text body of the message."
    assert e["received_at"] == "2026-08-16T10:00:00+00:00"

    # account was persisted so a second call reuses it instead of creating
    # a second disposable inbox
    assert agent.MAILTM_ACCOUNT_FILE.exists()
    saved = json.loads(agent.MAILTM_ACCOUNT_FILE.read_text())
    assert saved["address"] == address


def test_fetch_new_emails_mailtm_falls_back_to_html_then_intro(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "MAILTM_ACCOUNT_FILE", tmp_path / "mailtm_account.json")
    agent.MAILTM_ACCOUNT_FILE.write_text(json.dumps({"address": "existing@example.test", "password": "pw"}))

    def fake_request(method, path, payload=None, token=None, timeout=15):
        if path == "/token":
            return {"token": "tok"}
        if path == "/messages":
            return {"hydra:member": [{"id": "m1"}]}
        if path == "/messages/m1":
            # no "text" field at all -- must fall back to stripped html
            return {
                "id": "m1",
                "from": {"address": "x@example.test"},
                "subject": "No plain text",
                "html": ["<div>Only <em>html</em> here</div>"],
                "intro": "Only html...",
                "createdAt": "2026-08-16T11:00:00+00:00",
            }
        raise AssertionError(f"unexpected call: {path}")

    monkeypatch.setattr(agent, "_mailtm_request", fake_request)

    emails, address = agent.fetch_new_emails_mailtm()
    assert address == "existing@example.test"  # reused the saved account, didn't create a new one
    assert emails[0]["body"] == "Only html here"


def test_get_or_create_mailtm_account_reuses_saved_credentials(tmp_path, monkeypatch):
    account_file = tmp_path / "mailtm_account.json"
    monkeypatch.setattr(agent, "MAILTM_ACCOUNT_FILE", account_file)
    account_file.write_text(json.dumps({"address": "already@there.test", "password": "pw"}))

    def fail_if_called(*a, **kw):
        raise AssertionError("should not call the mail.tm API when an account is already saved")

    monkeypatch.setattr(agent, "_mailtm_request", fail_if_called)

    account = agent.get_or_create_mailtm_account()
    assert account == {"address": "already@there.test", "password": "pw"}
