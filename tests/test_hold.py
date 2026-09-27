"""Unit and flow tests for slew position hold, drift correction and unlock."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cranefleet.audit import Audit
from cranefleet.console import Console, Request
from cranefleet.errors import HoldNotEngaged
from cranefleet.hold import (
    DEFAULT_TOLERANCE,
    KIND_CORRECTION,
    KIND_ENGAGE,
    KIND_SAMPLE,
    Hold,
    confirm,
    disengage,
    engage,
    is_locked,
    persist_hold,
    restore_hold,
    track,
)
from cranefleet.slew import Slew
from cranefleet.store import MemoryStore
from cranefleet.wiring import build_app

WEB_ROOT = Path(__file__).resolve().parents[1] / "web"


def dispatch(console: Console, method: str, path: str, body: dict | None = None):  # noqa: ANN201
    request = Request(method=method, path=path, query={}, body=body or {}, body_valid=True)
    return console.dispatch(request)


class HoldUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hold = Hold("crane-1")
        self.slew = Slew("crane-1")
        self.audit = Audit(50)

    def test_engage_remembers_target(self) -> None:
        engage(self.hold, 30.0, recorder=self.audit)
        self.assertTrue(self.hold.is_engaged())
        self.assertEqual(self.hold.target_now(), 30.0)
        self.assertEqual(self.hold.events_snapshot()[0].kind, KIND_ENGAGE)
        self.assertIsNotNone(self.audit.latest("hold-engage"))

    def test_sample_within_tolerance_is_accepted(self) -> None:
        engage(self.hold, 30.0, recorder=self.audit)
        result = track(self.hold, self.slew, 31.0, self.audit)
        self.assertAlmostEqual(result.drift, 1.0)
        self.assertFalse(result.corrected)
        self.assertFalse(result.locked)
        self.assertAlmostEqual(self.slew.angle_now(), 31.0)
        self.assertEqual(self.hold.events_snapshot(1)[0].kind, KIND_SAMPLE)

    def test_drift_beyond_tolerance_corrects_and_locks(self) -> None:
        engage(self.hold, 30.0, recorder=self.audit)
        result = track(self.hold, self.slew, 33.0, self.audit)
        self.assertTrue(result.corrected)
        self.assertTrue(result.locked)
        self.assertAlmostEqual(result.angle, 30.0)
        # the boom is driven back to the held target
        self.assertAlmostEqual(self.slew.angle_now(), 30.0)
        self.assertEqual(self.hold.correction_count(), 1)
        self.assertAlmostEqual(self.hold.peak_drift_now(), 3.0)
        event = self.hold.events_snapshot(1)[0]
        self.assertEqual(event.kind, KIND_CORRECTION)
        self.assertAlmostEqual(event.drift, 3.0)
        self.assertIsNotNone(self.audit.latest("hold-correct"))

    def test_negative_drift_beyond_tolerance_also_corrects(self) -> None:
        engage(self.hold, 10.0, recorder=self.audit)
        result = track(self.hold, self.slew, 7.0, self.audit)
        self.assertTrue(result.corrected)
        self.assertAlmostEqual(result.drift, -3.0)
        self.assertAlmostEqual(self.slew.angle_now(), 10.0)

    def test_track_without_engage_is_rejected(self) -> None:
        with self.assertRaises(HoldNotEngaged):
            track(self.hold, self.slew, 30.0, self.audit)

    def test_locked_hold_keeps_target_until_confirmation(self) -> None:
        engage(self.hold, 30.0, recorder=self.audit)
        track(self.hold, self.slew, 33.0, self.audit)
        # a repeated stop cannot re-arm the hold at the drifted angle
        engage(self.hold, 99.0, recorder=self.audit)
        self.assertEqual(self.hold.target_now(), 30.0)
        self.assertTrue(is_locked(self.hold))
        self.assertTrue(confirm(self.hold, self.audit))
        self.assertFalse(self.hold.is_locked())
        # confirming twice is a harmless no-op
        self.assertFalse(confirm(self.hold, self.audit))

    def test_disengage_only_when_unlocked(self) -> None:
        engage(self.hold, 30.0, recorder=self.audit)
        self.assertTrue(disengage(self.hold, self.audit))
        self.assertFalse(self.hold.is_engaged())
        engage(self.hold, 30.0, recorder=self.audit)
        track(self.hold, self.slew, 33.0, self.audit)
        self.assertFalse(disengage(self.hold, self.audit))
        self.assertTrue(self.hold.is_engaged())

    def test_engage_accepts_custom_tolerance(self) -> None:
        engage(self.hold, 0.0, tolerance=0.5, recorder=self.audit)
        self.assertEqual(self.hold.tolerance_now(), 0.5)
        result = track(self.hold, self.slew, 0.4, self.audit)
        self.assertFalse(result.corrected)
        result = track(self.hold, self.slew, 1.0, self.audit)
        self.assertTrue(result.corrected)

    def test_persist_round_trip(self) -> None:
        store = MemoryStore()
        engage(self.hold, 45.0, tolerance=2.0, recorder=self.audit)
        track(self.hold, self.slew, 47.5, self.audit)
        self.assertTrue(self.hold.is_locked())
        persist_hold(self.hold, store)

        restored = Hold("crane-1")
        restore_hold(restored, store)
        self.assertTrue(restored.is_engaged())
        self.assertTrue(restored.is_locked())
        self.assertEqual(restored.target_now(), 45.0)
        self.assertEqual(restored.tolerance_now(), 2.0)
        self.assertEqual(restored.correction_count(), 1)
        kinds = [event.kind for event in restored.events_snapshot(50)]
        self.assertIn(KIND_CORRECTION, kinds)
        self.assertAlmostEqual(restored.peak_drift_now(), 2.5)


class HoldFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = build_app("memory", web_root=str(WEB_ROOT))
        self.console = self.app.console

    def tearDown(self) -> None:
        self.app.shutdown()

    def test_stop_engages_hold_automatically(self) -> None:
        status, payload = dispatch(self.console, "POST", "/api/slew/stop")
        self.assertEqual(status, 200)
        self.assertTrue(payload["data"]["hold_engaged"])
        status, payload = dispatch(self.console, "GET", "/api/hold/state")
        self.assertTrue(payload["data"]["engaged"])
        self.assertEqual(payload["data"]["target"], 0.0)
        self.assertEqual(payload["data"]["tolerance"], DEFAULT_TOLERANCE)

    def test_full_drift_correction_confirm_resume_flow(self) -> None:
        dispatch(self.console, "POST", "/api/slew/start")
        dispatch(self.console, "POST", "/api/slew/stop")
        # small drift inside the tolerance: no correction, no lock
        status, payload = dispatch(self.console, "POST", "/api/hold/track", {"angle": 0.8})
        self.assertEqual(status, 200)
        self.assertFalse(payload["data"]["track"]["corrected"])
        self.assertFalse(payload["data"]["track"]["locked"])
        # wind pushes the boom beyond the allowed range
        status, payload = dispatch(self.console, "POST", "/api/hold/track", {"angle": 3.0})
        self.assertEqual(status, 200)
        self.assertTrue(payload["data"]["track"]["corrected"])
        self.assertTrue(payload["data"]["hold"]["locked"])
        self.assertEqual(payload["data"]["hold"]["angle"], 0.0)
        self.assertEqual(payload["data"]["hold"]["corrections"], 1)
        # slewing stays blocked until on-site confirmation
        status, payload = dispatch(self.console, "POST", "/api/slew/start")
        self.assertEqual(status, 409)
        self.assertIn("confirmation", payload["error"])
        # the crane-command swing path is blocked as well while locked
        status, payload = dispatch(
            self.console,
            "POST",
            "/api/crane/command",
            {"source": "cab", "action": "swing", "angle": 20},
        )
        self.assertEqual(status, 409)
        # on-site confirmation releases the lock but keeps the hold
        status, payload = dispatch(self.console, "POST", "/api/hold/confirm")
        self.assertEqual(status, 200)
        self.assertTrue(payload["data"]["released"])
        self.assertTrue(payload["data"]["hold"]["engaged"])
        self.assertFalse(payload["data"]["hold"]["locked"])
        # a new slew is allowed again and drops the hold
        status, payload = dispatch(self.console, "POST", "/api/slew/start")
        self.assertEqual(status, 200)
        self.assertFalse(payload["data"]["hold_engaged"])

    def test_track_without_hold_returns_409(self) -> None:
        status, payload = dispatch(self.console, "POST", "/api/hold/track", {"angle": 5.0})
        self.assertEqual(status, 409)
        self.assertFalse(payload["ok"])

    def test_track_validates_angle(self) -> None:
        dispatch(self.console, "POST", "/api/slew/stop")
        status, _ = dispatch(self.console, "POST", "/api/hold/track", {})
        self.assertEqual(status, 400)
        request = Request("POST", "/api/hold/track", {}, {}, body_valid=False)
        status, _ = self.console.dispatch(request)
        self.assertEqual(status, 400)

    def test_manual_engage_with_tolerance(self) -> None:
        status, payload = dispatch(
            self.console, "POST", "/api/hold/engage", {"angle": 12.0, "tolerance": 2.0}
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["data"]["target"], 12.0)
        self.assertEqual(payload["data"]["tolerance"], 2.0)
        status, payload = dispatch(self.console, "POST", "/api/hold/engage", {"tolerance": -1})
        self.assertEqual(status, 400)

    def test_events_endpoint_lists_drift_and_corrections_newest_first(self) -> None:
        dispatch(self.console, "POST", "/api/slew/stop")
        dispatch(self.console, "POST", "/api/hold/track", {"angle": 0.5})
        dispatch(self.console, "POST", "/api/hold/track", {"angle": -3.0})
        dispatch(self.console, "POST", "/api/hold/confirm")
        status, payload = dispatch(self.console, "GET", "/api/hold/events")
        self.assertEqual(status, 200)
        kinds = [event["kind"] for event in payload["data"]]
        self.assertEqual(kinds[0], "confirm")
        self.assertIn("correction", kinds)
        self.assertIn("sample", kinds)
        self.assertIn("engage", kinds)
        correction = next(event for event in payload["data"] if event["kind"] == "correction")
        self.assertAlmostEqual(correction["drift"], -3.0)
        # the significant events also land in the audit trail
        actions = [entry.action for entry in self.console.audit.snapshot()]
        self.assertIn("hold-engage", actions)
        self.assertIn("hold-correct", actions)
        self.assertIn("hold-confirm", actions)

    def test_cab_stop_command_engages_hold(self) -> None:
        dispatch(self.console, "POST", "/api/slew/start")
        dispatch(
            self.console,
            "POST",
            "/api/crane/command",
            {"source": "cab", "action": "stop", "axis": "slew"},
        )
        status, payload = dispatch(self.console, "GET", "/api/hold/state")
        self.assertTrue(payload["data"]["engaged"])

    def test_state_view_includes_hold(self) -> None:
        status, payload = dispatch(self.console, "GET", "/api/state")
        self.assertEqual(status, 200)
        self.assertIn("hold", payload["data"])
        self.assertIn("recent_events", payload["data"]["hold"])

    def test_locked_hold_survives_a_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = build_app(directory, web_root=str(WEB_ROOT))
            dispatch(first.console, "POST", "/api/slew/stop")
            dispatch(first.console, "POST", "/api/hold/track", {"angle": 4.0})
            self.assertTrue(first.console.hold.is_locked())
            first.shutdown()

            second = build_app(directory, web_root=str(WEB_ROOT))
            self.assertTrue(second.console.hold.is_locked())
            self.assertEqual(second.console.hold.correction_count(), 1)
            status, payload = dispatch(second.console, "POST", "/api/slew/start")
            self.assertEqual(status, 409)
            status, payload = dispatch(second.console, "GET", "/api/hold/events")
            self.assertIn(
                "correction", [event["kind"] for event in payload["data"]]
            )
            second.shutdown()


if __name__ == "__main__":
    unittest.main()
