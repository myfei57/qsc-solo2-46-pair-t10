"""Unit and integration tests for the anti-drift position hold."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cranefleet.audit import Audit
from cranefleet.console import Console, Request
from cranefleet.errors import (
    HoldAlreadyEngaged,
    HoldLocked,
    HoldNotEngaged,
    HoldNotLocked,
    HoldSlewMoving,
    SlewBlocked,
)
from cranefleet.geometry import Point
from cranefleet.hold import (
    DEFAULT_TOLERANCE,
    EVENT_CONFIRM,
    EVENT_CORRECT,
    EVENT_DRIFT,
    EVENT_ENGAGE,
    EVENT_RELEASE,
    STATE_HOLDING,
    STATE_INACTIVE,
    STATE_LOCKED,
    Hold,
    confirm,
    drift,
    engage,
    events_snapshot,
    is_active,
    is_locked,
    persist,
    release,
    restore,
)
from cranefleet.slew import Slew
from cranefleet.slew import start as slew_start
from cranefleet.store import MemoryStore
from cranefleet.wiring import build_app
from cranefleet.zone import ZoneState

BOUNDARY = [Point(0.0, 0.0), Point(100.0, 0.0), Point(100.0, 100.0), Point(0.0, 100.0)]
WEB_ROOT = Path(__file__).resolve().parents[1] / "web"


def dispatch(console: Console, method: str, path: str, body: dict | None = None, query: dict | None = None):  # noqa: ANN201
    request = Request(method=method, path=path, query=query or {}, body=body or {}, body_valid=True)
    return console.dispatch(request)


class HoldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.slew = Slew("crane-1")
        self.hold = Hold("crane-1")
        self.zone = ZoneState("zone-1", "zone", BOUNDARY)
        self.store = MemoryStore()
        self.audit = Audit(100)

    def engage_hold(self) -> float:
        return engage(self.hold, self.slew, DEFAULT_TOLERANCE, self.audit)

    def test_engage_requires_the_slew_at_rest(self) -> None:
        slew_start(self.slew, self.zone, self.store, self.audit)
        with self.assertRaises(HoldSlewMoving):
            self.engage_hold()
        self.assertEqual(self.hold.state_now(), STATE_INACTIVE)
        self.assertFalse(is_active(self.hold))

    def test_engage_locks_the_slew_on_target(self) -> None:
        held = self.engage_hold()
        self.assertEqual(held, 0.0)
        self.assertEqual(self.hold.state_now(), STATE_HOLDING)
        self.assertTrue(is_active(self.hold))
        self.assertFalse(is_locked(self.hold))
        self.assertTrue(self.slew.held_by_hold())
        with self.assertRaises(SlewBlocked):
            slew_start(self.slew, self.zone, self.store, self.audit)
        kinds = [event.kind for event in events_snapshot(self.hold)]
        self.assertEqual(kinds, [EVENT_ENGAGE])

    def test_engage_at_an_explicit_target(self) -> None:
        held = engage(self.hold, self.slew, DEFAULT_TOLERANCE, self.audit, target=45.0)
        self.assertEqual(held, 45.0)
        self.assertEqual(self.hold.target_now(), 45.0)

    def test_drift_within_tolerance_is_only_tracked(self) -> None:
        self.engage_hold()
        result = drift(self.hold, self.slew, 0.3, self.audit)
        self.assertFalse(result.corrected)
        self.assertFalse(result.locked)
        self.assertAlmostEqual(result.deviation, 0.3)
        self.assertAlmostEqual(self.slew.angle_now(), 0.3)
        self.assertEqual(self.hold.drift_observations_now(), 1)
        self.assertAlmostEqual(self.hold.last_drift_now(), 0.3)
        self.assertAlmostEqual(self.hold.peak_drift_now(), 0.3)
        self.assertEqual(self.hold.event_count(), 1)

    def test_drift_past_tolerance_is_corrected_and_locked(self) -> None:
        self.engage_hold()
        drift(self.hold, self.slew, 0.3, self.audit)
        result = drift(self.hold, self.slew, 0.4, self.audit)
        self.assertTrue(result.corrected)
        self.assertTrue(result.locked)
        self.assertAlmostEqual(result.deviation, 0.7)
        self.assertAlmostEqual(self.slew.angle_now(), 0.0)
        self.assertEqual(self.hold.state_now(), STATE_LOCKED)
        self.assertTrue(is_locked(self.hold))
        self.assertTrue(self.hold.has_alert())
        self.assertEqual(self.hold.correction_count_now(), 1)
        kinds = [event.kind for event in events_snapshot(self.hold)]
        self.assertEqual(kinds, [EVENT_CORRECT, EVENT_DRIFT, EVENT_ENGAGE])
        actions = [entry.action for entry in self.audit.snapshot()]
        self.assertIn(EVENT_DRIFT, actions)
        self.assertIn(EVENT_CORRECT, actions)

    def test_locked_hold_rejects_release_drift_and_start(self) -> None:
        self.engage_hold()
        drift(self.hold, self.slew, 1.0, self.audit)
        with self.assertRaises(HoldLocked):
            release(self.hold, self.slew, self.audit)
        with self.assertRaises(HoldLocked):
            drift(self.hold, self.slew, 0.1, self.audit)
        with self.assertRaises(SlewBlocked):
            slew_start(self.slew, self.zone, self.store, self.audit)

    def test_confirm_then_release_unlocks_the_slew(self) -> None:
        self.engage_hold()
        drift(self.hold, self.slew, 1.0, self.audit)
        confirm(self.hold, "老张", self.audit)
        self.assertEqual(self.hold.state_now(), STATE_HOLDING)
        self.assertFalse(self.hold.has_alert())
        self.assertEqual(self.hold.confirmed_by_now(), "老张")
        self.assertTrue(self.slew.held_by_hold())
        release(self.hold, self.slew, self.audit)
        self.assertEqual(self.hold.state_now(), STATE_INACTIVE)
        self.assertFalse(self.slew.held_by_hold())
        slew_start(self.slew, self.zone, self.store, self.audit)
        self.assertTrue(self.slew.is_moving())
        kinds = [event.kind for event in events_snapshot(self.hold)]
        self.assertEqual(
            kinds,
            [EVENT_RELEASE, EVENT_CONFIRM, EVENT_CORRECT, EVENT_DRIFT, EVENT_ENGAGE],
        )

    def test_drift_after_confirm_locks_again(self) -> None:
        self.engage_hold()
        drift(self.hold, self.slew, 1.0, self.audit)
        confirm(self.hold, "", self.audit)
        result = drift(self.hold, self.slew, 0.8, self.audit)
        self.assertTrue(result.corrected)
        self.assertEqual(self.hold.correction_count_now(), 2)

    def test_idle_and_duplicate_operations_are_rejected(self) -> None:
        with self.assertRaises(HoldNotEngaged):
            drift(self.hold, self.slew, 0.1, self.audit)
        with self.assertRaises(HoldNotEngaged):
            release(self.hold, self.slew, self.audit)
        with self.assertRaises(HoldNotLocked):
            confirm(self.hold, "", self.audit)
        self.engage_hold()
        with self.assertRaises(HoldAlreadyEngaged):
            self.engage_hold()

    def test_persist_restore_round_trip(self) -> None:
        self.engage_hold()
        drift(self.hold, self.slew, 1.0, self.audit)
        persist(self.hold, self.store)
        reloaded = Hold("crane-1")
        restore(reloaded, self.store)
        self.assertEqual(reloaded.state_now(), STATE_LOCKED)
        self.assertTrue(reloaded.has_alert())
        self.assertEqual(reloaded.target_now(), 0.0)
        self.assertEqual(reloaded.tolerance_now(), DEFAULT_TOLERANCE)
        self.assertEqual(reloaded.correction_count_now(), 1)
        self.assertEqual(reloaded.event_count(), 3)
        missing = Hold("crane-9")
        restore(missing, self.store)
        self.assertEqual(missing.state_now(), STATE_INACTIVE)


class HoldApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = build_app("memory", web_root=str(WEB_ROOT))
        self.console = self.app.console

    def tearDown(self) -> None:
        self.app.shutdown()

    def test_full_hold_flow(self) -> None:
        status, payload = dispatch(self.console, "POST", "/api/hold/engage")
        self.assertEqual(status, 200)
        self.assertEqual(payload["data"]["state"], "holding")
        self.assertEqual(payload["data"]["target"], 0.0)

        status, payload = dispatch(self.console, "POST", "/api/slew/start")
        self.assertEqual(status, 409)
        self.assertEqual(payload["error"], "position hold active")

        status, payload = dispatch(self.console, "POST", "/api/hold/drift", {"angle": 0.3})
        self.assertEqual(status, 200)
        self.assertFalse(payload["data"]["drift"]["corrected"])
        self.assertEqual(payload["data"]["hold"]["state"], "holding")

        status, payload = dispatch(self.console, "POST", "/api/hold/drift", {"angle": 0.4})
        self.assertEqual(status, 200)
        self.assertTrue(payload["data"]["drift"]["corrected"])
        self.assertTrue(payload["data"]["drift"]["locked"])
        self.assertEqual(payload["data"]["hold"]["state"], "locked")
        self.assertTrue(payload["data"]["hold"]["alert"])
        self.assertEqual(payload["data"]["hold"]["correction_count"], 1)
        self.assertEqual(payload["data"]["hold"]["angle"], 0.0)

        status, payload = dispatch(self.console, "GET", "/api/hold/events")
        self.assertEqual(status, 200)
        kinds = [event["kind"] for event in payload["data"]]
        self.assertEqual(kinds, ["hold-correct", "hold-drift", "hold-engage"])

        status, payload = dispatch(self.console, "POST", "/api/hold/release")
        self.assertEqual(status, 409)
        self.assertEqual(payload["error"], "correction pending on-site confirmation")
        status, payload = dispatch(self.console, "POST", "/api/slew/start")
        self.assertEqual(status, 409)
        self.assertEqual(payload["error"], "correction pending on-site confirmation")

        status, payload = dispatch(self.console, "POST", "/api/hold/confirm", {"operator": "李工"})
        self.assertEqual(status, 200)
        self.assertEqual(payload["data"]["state"], "holding")
        self.assertFalse(payload["data"]["alert"])
        self.assertEqual(payload["data"]["confirmed_by"], "李工")

        status, payload = dispatch(self.console, "POST", "/api/hold/release")
        self.assertEqual(status, 200)
        self.assertEqual(payload["data"]["state"], "inactive")

        status, payload = dispatch(self.console, "POST", "/api/slew/start")
        self.assertEqual(status, 200)
        self.assertTrue(payload["data"]["moving"])

        status, payload = dispatch(self.console, "GET", "/api/audit", query={"action": ["hold-correct"]})
        self.assertEqual(status, 200)
        self.assertEqual(len(payload["data"]), 1)

    def test_engage_while_moving_is_rejected(self) -> None:
        dispatch(self.console, "POST", "/api/slew/start")
        status, payload = dispatch(self.console, "POST", "/api/hold/engage")
        self.assertEqual(status, 409)
        self.assertEqual(payload["error"], "slew still moving")

    def test_drift_and_release_require_an_engaged_hold(self) -> None:
        status, payload = dispatch(self.console, "POST", "/api/hold/drift", {"angle": 1.0})
        self.assertEqual(status, 409)
        self.assertEqual(payload["error"], "hold not engaged")
        status, payload = dispatch(self.console, "POST", "/api/hold/release")
        self.assertEqual(status, 409)
        self.assertEqual(payload["error"], "hold not engaged")
        status, payload = dispatch(self.console, "POST", "/api/hold/confirm")
        self.assertEqual(status, 409)
        self.assertEqual(payload["error"], "no correction awaiting confirmation")

    def test_tolerance_comes_from_the_site_config(self) -> None:
        status, _ = dispatch(
            self.console,
            "POST",
            "/api/config",
            {"site_name": "华东三号工地", "safe_separation": 12, "hold_tolerance": 2.0},
        )
        self.assertEqual(status, 200)
        dispatch(self.console, "POST", "/api/hold/engage")
        status, payload = dispatch(self.console, "POST", "/api/hold/drift", {"angle": 1.5})
        self.assertFalse(payload["data"]["drift"]["corrected"])
        status, payload = dispatch(self.console, "POST", "/api/hold/drift", {"angle": 1.0})
        self.assertTrue(payload["data"]["drift"]["corrected"])

    def test_state_snapshot_exposes_the_hold(self) -> None:
        status, payload = dispatch(self.console, "GET", "/api/state")
        self.assertEqual(status, 200)
        self.assertIn("hold", payload["data"])
        self.assertEqual(payload["data"]["hold"]["state"], "inactive")
        self.assertFalse(payload["data"]["slew"]["hold_locked"])
        status, payload = dispatch(self.console, "GET", "/api/config")
        self.assertEqual(payload["data"]["hold_tolerance"], 0.5)


class HoldBootTests(unittest.TestCase):
    def test_locked_hold_survives_a_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = build_app(directory, web_root=str(WEB_ROOT))
            dispatch(first.console, "POST", "/api/hold/engage")
            dispatch(first.console, "POST", "/api/hold/drift", {"angle": 1.2})
            first.shutdown()

            second = build_app(directory, web_root=str(WEB_ROOT))
            self.assertEqual(second.console.hold.state_now(), STATE_LOCKED)
            self.assertTrue(second.console.slew.held_by_hold())
            status, payload = dispatch(second.console, "POST", "/api/slew/start")
            self.assertEqual(status, 409)
            self.assertEqual(payload["error"], "correction pending on-site confirmation")
            status, payload = dispatch(second.console, "GET", "/api/hold/events")
            kinds = [event["kind"] for event in payload["data"]]
            self.assertIn("hold-correct", kinds)
            dispatch(second.console, "POST", "/api/hold/confirm", {"operator": "王工"})
            dispatch(second.console, "POST", "/api/hold/release")
            status, payload = dispatch(second.console, "POST", "/api/slew/start")
            self.assertEqual(status, 200)
            self.assertTrue(payload["data"]["moving"])
            second.shutdown()


if __name__ == "__main__":
    unittest.main()
