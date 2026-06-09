"""API regressions for the live-trading runtime wiring (R6: C1 + C2 + runner control).

Covers the open-file integration seam the orchestrator and parcel R6 own:
- C1: a ``propose_mandate_profiles`` tool_result is translated into a top-level
  ``mandate.proposal`` SSE frame WITHOUT touching the protected ``src/agent/loop.py``.
- C2: ``GET /live/status`` surfaces the dormant-by-default channel (auth, mandate,
  runner liveness, halt) and ``POST /live/authorize`` is a discover-only on-ramp.
- Runner control: ``POST /live/runner/start|stop`` are privileged surface actions
  gated on a committed, unexpired mandate and a clear kill switch.

All tests run against stubbed runner/liveness state — no real agent or broker.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

import api_server


def _client(tmp_path: Path, monkeypatch) -> TestClient:
    # Redirect the runtime root (``~/.vibe-trading``) at the home boundary so the
    # live tree, HALT sentinel, mandate store, and proposal store all resolve
    # under tmp_path. get_runtime_root() == Path.home() / ".vibe-trading".
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path), raising=False)
    monkeypatch.setattr(api_server, "_runner_tasks", {}, raising=False)
    monkeypatch.setattr(api_server, "_runner_factory", None, raising=False)
    return TestClient(api_server.app, client=("127.0.0.1", 50000))


def _valid_mandate_state(broker: str = "robinhood") -> api_server.ActiveMandateState:
    """Build a committed, unexpired active-mandate snapshot for stubbing."""
    return api_server.ActiveMandateState(
        broker=broker,
        account_ref="acct_test",
        created_at="2026-05-29T00:00:00+00:00",
        expires_at="2099-01-01T00:00:00+00:00",
        expires_in_seconds=10_000_000,
        expired=False,
        limits=api_server.MandateLimits(
            max_order_notional_usd=750.0,
            max_total_exposure_usd=5000.0,
            max_leverage=1.0,
            max_trades_per_day=5,
            allowed_instruments=["equity"],
            account_funding_usd=5000.0,
        ),
    )


# --------------------------------------------------------------------------- #
# C2 — GET /live/status
# --------------------------------------------------------------------------- #


def test_live_status_dormant_by_default(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.get("/live/status")

    assert response.status_code == 200
    body = response.json()
    assert body["global_halted"] is False
    # No live MCP brokers configured in the India-only fork
    assert body["brokers"] == []


def test_live_status_unknown_broker_not_live(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.get("/live/status", params={"broker": "unknown_broker"})

    assert response.status_code == 200
    body = response.json()
    assert len(body["brokers"]) == 1
    assert body["brokers"][0]["auth"]["is_live_broker"] is False


def test_live_status_blank_broker_rejected(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.get("/live/status", params={"broker": "   "})

    assert response.status_code == 400


# --------------------------------------------------------------------------- #
# C2 — POST /live/authorize (discover-only on-ramp; never authorizes server-side)
# --------------------------------------------------------------------------- #


def test_authorize_unknown_broker_rejected(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    # No live MCP brokers configured in the India-only fork
    response = client.post("/live/authorize", json={"broker": "etrade"})
    assert response.status_code == 400

    response = client.post("/live/authorize", json={"broker": "robinhood"})
    assert response.status_code == 400


# --------------------------------------------------------------------------- #
# Runner control — POST /live/runner/start|stop
# --------------------------------------------------------------------------- #


def test_runner_start_unknown_broker_rejected(tmp_path: Path, monkeypatch) -> None:
    """No live MCP brokers configured — runner start is rejected."""
    client = _client(tmp_path, monkeypatch)

    response = client.post("/live/runner/start", json={"broker": "robinhood"})

    # robinhood is no longer a known live broker
    assert response.status_code in (400, 409)


def test_runner_stop_rejects_unknown_broker(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    # No live runner profiles configured for robinhood in India-only fork
    response = client.post("/live/runner/stop", json={"broker": "robinhood"})

    assert response.status_code == 400


# --------------------------------------------------------------------------- #
# C1 — propose_mandate_profiles tool_result -> mandate.proposal SSE frame
# --------------------------------------------------------------------------- #


def _seed_proposal(tmp_path: Path, proposal_id: str, broker: str = "robinhood") -> dict:
    proposal = {
        "type": "mandate.proposal",
        "proposal_id": proposal_id,
        "session_id": "s1",
        "profiles": [{"ordinal": 1, "label": "稳健", "max_order_usd": 250}],
    }
    proposals_dir = tmp_path / ".vibe-trading" / "live" / broker / "proposals"
    proposals_dir.mkdir(parents=True, exist_ok=True)
    (proposals_dir / f"{proposal_id}.json").write_text(
        json.dumps(proposal), encoding="utf-8"
    )
    return proposal


def test_c1_relay_builds_mandate_proposal_frame(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path), raising=False)
    proposal_id = "mp_01ABCdef"
    _seed_proposal(tmp_path, proposal_id)

    event = SimpleNamespace(
        event_type="tool_result",
        session_id="s1",
        data={
            "tool": "propose_mandate_profiles",
            "status": "ok",
            "preview": json.dumps({"proposal_id": proposal_id})[:200],
        },
    )

    frame = api_server._mandate_proposal_frame_from_tool_result(event)

    assert frame is not None
    assert "mandate.proposal" in frame
    assert proposal_id in frame


def test_c1_relay_ignores_non_propose_events(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path), raising=False)

    # A different tool's result: no relay.
    other = SimpleNamespace(
        event_type="tool_result",
        session_id="s1",
        data={"tool": "run_backtest", "status": "ok", "preview": "{}"},
    )
    assert api_server._mandate_proposal_frame_from_tool_result(other) is None

    # A non-tool_result event: no relay.
    thinking = SimpleNamespace(event_type="thinking", session_id="s1", data={})
    assert api_server._mandate_proposal_frame_from_tool_result(thinking) is None


def test_c1_relay_returns_none_when_proposal_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path), raising=False)
    # Preview references an id with no persisted proposal on disk.
    event = SimpleNamespace(
        event_type="tool_result",
        session_id="s1",
        data={
            "tool": "propose_mandate_profiles",
            "status": "ok",
            "preview": json.dumps({"proposal_id": "mp_missing01"})[:200],
        },
    )
    assert api_server._mandate_proposal_frame_from_tool_result(event) is None


# --------------------------------------------------------------------------- #
# Real runner factory (R-INT): no more TypeError on the default path
# --------------------------------------------------------------------------- #


def test_build_live_runner_no_broker_configured_raises_unavailable(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path), raising=False)
    monkeypatch.setattr(api_server, "_runner_factory", None, raising=False)
    # No robinhood MCP server in the default config → clean 503-class error,
    # NOT a TypeError (the audit's CRITICAL finding).
    import pytest

    with pytest.raises(api_server.LiveRunnerUnavailable):
        api_server._build_live_runner("robinhood")


def test_build_live_runner_raises_for_removed_broker(tmp_path, monkeypatch) -> None:
    """Robinhood runner wiring removed in India-only fork — raises unavailable."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path), raising=False)
    monkeypatch.setattr(api_server, "_runner_factory", None, raising=False)

    class _StubAdapter:
        def call_tool(self, name, args):
            return {"status": "ok", "result": {}}

    monkeypatch.setattr(api_server, "_live_broker_adapter", lambda broker: _StubAdapter())

    import pytest
    with pytest.raises(api_server.LiveRunnerUnavailable):
        api_server._build_live_runner("robinhood")


def test_runner_start_rejects_unknown_broker(tmp_path, monkeypatch) -> None:
    """Runner start returns 400 for brokers with no live runner profile."""
    client = _client(tmp_path, monkeypatch)

    response = client.post("/live/runner/start", json={"broker": "robinhood"})
    assert response.status_code == 400


# --------------------------------------------------------------------------- #
# H5 — per-order live.action SSE relay (reload full record from the ledger)
# --------------------------------------------------------------------------- #


def _seed_ledger(tmp_path: Path, record: dict) -> None:
    ledger = tmp_path / ".vibe-trading" / "live" / "audit.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def test_live_action_relay_builds_frame_from_guard_result(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path), raising=False)
    audit_id = "la_01abcDEF"
    _seed_ledger(tmp_path, {"audit_id": audit_id, "kind": "order_placed", "outcome": "accepted",
                            "broker_request": {"symbol": "NVDA"}})

    event = SimpleNamespace(
        event_type="tool_result",
        session_id="s1",
        data={
            "tool": "mcp_robinhood_place_order",
            "status": "ok",
            "preview": json.dumps({"status": "ok", "live_action": {"audit_id": audit_id}})[:200],
        },
    )

    frame = api_server._live_action_frame_from_tool_result(event)
    assert frame is not None
    assert "live.action" in frame
    assert audit_id in frame and "order_placed" in frame


def test_live_action_relay_ignores_non_live_results(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path), raising=False)
    # A normal tool_result with no live_action marker → no relay.
    event = SimpleNamespace(
        event_type="tool_result", session_id="s1",
        data={"tool": "run_backtest", "status": "ok", "preview": "{}"},
    )
    assert api_server._live_action_frame_from_tool_result(event) is None


# --------------------------------------------------------------------------- #
# H9 — broker-derived ceilings hook
# --------------------------------------------------------------------------- #


def test_fetch_broker_ceilings_derives_from_account(tmp_path, monkeypatch) -> None:
    class _StubAdapter:
        def call_tool(self, name, args):
            assert name == "get_account"
            return {"status": "ok", "result": {"buying_power": 4200.0}}

    monkeypatch.setattr(api_server, "_live_broker_adapter", lambda broker: _StubAdapter())
    ceilings = api_server._fetch_broker_ceilings("robinhood")
    assert ceilings == {
        "account_funding_usd": 4200.0,
        "max_order_notional_usd": 4200.0,
        "max_total_exposure_usd": 4200.0,
    }


def test_fetch_broker_ceilings_falls_back_to_none(tmp_path, monkeypatch) -> None:
    # Unavailable broker → None (commit falls back to the proposal snapshot).
    def _unavail(broker):
        raise api_server.LiveRunnerUnavailable("x")

    monkeypatch.setattr(api_server, "_live_broker_adapter", _unavail)
    assert api_server._fetch_broker_ceilings("robinhood") is None
