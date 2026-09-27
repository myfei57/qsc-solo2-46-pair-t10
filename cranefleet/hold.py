"""Slew position hold: engage on arrival, track wind drift, auto-correct.

Once the slew reaches its working position the hold controller keeps the boom
on the target angle. Wind still nudges the boom, so every encoder sample is
checked against the allowed drift tolerance. A sample outside the tolerance
triggers an automatic correction back to the target, raises an alert and locks
further slewing until an on-site confirmation releases the lock. Every sample
and correction is kept in a bounded, persisted log so the drift and correction
history can be reviewed after the fact.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from .audit import Audit
from .errors import HoldNotEngaged, StoreNotFound
from .slew import Slew, align as slew_align
from .store import Store

DEFAULT_TOLERANCE = 1.5
EVENT_CAPACITY = 200
HOLD_KEY = "hold/crane-1"

KIND_ENGAGE = "engage"
KIND_SAMPLE = "sample"
KIND_CORRECTION = "correction"
KIND_CONFIRM = "confirm"
KIND_DISENGAGE = "disengage"

ACTION_ENGAGE = "hold-engage"
ACTION_CORRECT = "hold-correct"
ACTION_CONFIRM = "hold-confirm"
ACTION_DISENGAGE = "hold-disengage"


@dataclass(frozen=True)
class HoldEvent:
    """One thing that happened while holding: a sample, correction or release."""

    sequence: int
    recorded_at: str
    kind: str
    angle: float
    drift: float

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "at": self.recorded_at,
            "kind": self.kind,
            "angle": self.angle,
            "drift": self.drift,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "HoldEvent":
        return cls(
            sequence=int(payload["sequence"]),
            recorded_at=str(payload["at"]),
            kind=str(payload["kind"]),
            angle=float(payload["angle"]),
            drift=float(payload["drift"]),
        )


@dataclass(frozen=True)
class TrackResult:
    """The outcome of one encoder sample against the held position."""

    angle: float
    target: float
    drift: float
    tolerance: float
    corrected: bool
    locked: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "angle": self.angle,
            "target": self.target,
            "drift": self.drift,
            "tolerance": self.tolerance,
            "corrected": self.corrected,
            "locked": self.locked,
        }


class Hold:
    """Mutable state of the position-hold controller for one crane."""

    def __init__(self, hold_id: str, tolerance: float = DEFAULT_TOLERANCE) -> None:
        self._lock = threading.Lock()
        self.id = hold_id
        self.engaged = False
        self.target = 0.0
        self.tolerance = tolerance
        self.drift = 0.0
        self.locked = False
        self.corrections = 0
        self.peak_drift = 0.0
        self.sequence = 0
        self.events: list[HoldEvent] = []

    def is_engaged(self) -> bool:
        with self._lock:
            return self.engaged

    def is_locked(self) -> bool:
        with self._lock:
            return self.locked

    def target_now(self) -> float:
        with self._lock:
            return self.target

    def tolerance_now(self) -> float:
        with self._lock:
            return self.tolerance

    def drift_now(self) -> float:
        with self._lock:
            return self.drift

    def correction_count(self) -> int:
        with self._lock:
            return self.corrections

    def peak_drift_now(self) -> float:
        with self._lock:
            return self.peak_drift

    def event_sequence(self) -> int:
        with self._lock:
            return self.sequence

    def events_snapshot(self, limit: int = 50) -> list[HoldEvent]:
        """Return the newest events first, up to ``limit`` entries."""

        with self._lock:
            events = list(self.events)
        if limit <= 0 or limit > len(events):
            limit = len(events)
        return list(reversed(events[-limit:]))

    def event_count(self) -> int:
        with self._lock:
            return len(self.events)


def is_locked(hold: Hold) -> bool:
    return hold.is_locked()


def engage(
    hold: Hold,
    angle: float,
    tolerance: float | None = None,
    recorder: Audit | None = None,
) -> None:
    """Arm the hold at the current angle after the slew reaches position.

    A hold that is locked pending on-site confirmation keeps its original
    target: a repeated stop cannot quietly re-arm and clear the lock.
    """

    with hold._lock:  # noqa: SLF001 - module-level functions share hold state
        if hold.locked:
            return
        if tolerance is not None and tolerance > 0:
            hold.tolerance = tolerance
        hold.engaged = True
        hold.target = angle
        hold.drift = 0.0
        hold.peak_drift = 0.0
        hold.sequence += 1
        hold.events.append(
            HoldEvent(hold.sequence, _now(), KIND_ENGAGE, angle, 0.0)
        )
        _trim(hold)
        target = hold.target
        bound = hold.tolerance
    if recorder is not None:
        recorder.record(
            hold.id,
            ACTION_ENGAGE,
            f"target={target:.2f} tolerance={bound:.2f}",
        )


def track(
    hold: Hold,
    slew: Slew,
    measured: float,
    recorder: Audit | None = None,
) -> TrackResult:
    """Accept one encoder sample and correct when the drift is out of range.

    The measured angle reflects wind pushing the boom. Inside the tolerance it
    is accepted as-is; outside it, the slew is driven back to the target, the
    correction is logged and the slew locks until on-site confirmation.
    """

    with hold._lock:  # noqa: SLF001
        if not hold.engaged:
            raise HoldNotEngaged("position hold is not engaged")
        drift = measured - hold.target
        hold.drift = drift
        if abs(drift) > abs(hold.peak_drift):
            hold.peak_drift = drift
        corrected = abs(drift) > hold.tolerance
        kind = KIND_CORRECTION if corrected else KIND_SAMPLE
        hold.sequence += 1
        hold.events.append(HoldEvent(hold.sequence, _now(), kind, measured, drift))
        if corrected:
            hold.corrections += 1
            hold.locked = True
            # the held error is back to zero once the correction lands
            hold.drift = 0.0
        _trim(hold)
        target = hold.target
        bound = hold.tolerance
        sequence = hold.sequence
        locked = hold.locked

    slew_align(slew, measured)
    if corrected:
        slew_align(slew, target)
        # the trigger drift is reported with the alert even though the held
        # error has already been driven back to zero
        if recorder is not None:
            recorder.record(
                hold.id,
                ACTION_CORRECT,
                f"seq={sequence} drift={drift:.2f} corrected_to={target:.2f}",
            )
        return TrackResult(target, target, drift, bound, True, locked)
    return TrackResult(measured, target, drift, bound, False, locked)


def confirm(hold: Hold, recorder: Audit | None = None) -> bool:
    """Release the correction lock after an on-site confirmation."""

    with hold._lock:  # noqa: SLF001
        was_locked = hold.locked
        if not was_locked:
            return False
        hold.locked = False
        hold.sequence += 1
        hold.events.append(
            HoldEvent(hold.sequence, _now(), KIND_CONFIRM, hold.target, hold.drift)
        )
        _trim(hold)
        target = hold.target
    if recorder is not None:
        recorder.record(hold.id, ACTION_CONFIRM, f"target={target:.2f} lock released")
    return True


def disengage(hold: Hold, recorder: Audit | None = None) -> bool:
    """Drop the hold when a new slew is started; a locked hold cannot drop."""

    with hold._lock:  # noqa: SLF001
        if not hold.engaged or hold.locked:
            return False
        hold.engaged = False
        hold.drift = 0.0
        hold.peak_drift = 0.0
        hold.sequence += 1
        hold.events.append(
            HoldEvent(hold.sequence, _now(), KIND_DISENGAGE, hold.target, 0.0)
        )
        target = hold.target
    if recorder is not None:
        recorder.record(hold.id, ACTION_DISENGAGE, f"target={target:.2f}")
    return True


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trim(hold: Hold) -> None:
    if len(hold.events) > EVENT_CAPACITY:
        del hold.events[: len(hold.events) - EVENT_CAPACITY]


def persist_hold(hold: Hold, store: Store) -> None:
    with hold._lock:  # noqa: SLF001
        payload = json.dumps(
            {
                "engaged": hold.engaged,
                "target": hold.target,
                "tolerance": hold.tolerance,
                "drift": hold.drift,
                "locked": hold.locked,
                "corrections": hold.corrections,
                "peak_drift": hold.peak_drift,
                "sequence": hold.sequence,
                "events": [event.to_dict() for event in hold.events],
            }
        ).encode("utf-8")
    store.save(HOLD_KEY, payload)


def restore_hold(hold: Hold, store: Store) -> None:
    try:
        payload = store.load(HOLD_KEY)
    except StoreNotFound:
        return
    snapshot = json.loads(payload.decode("utf-8"))
    with hold._lock:  # noqa: SLF001
        hold.engaged = bool(snapshot.get("engaged", False))
        hold.target = float(snapshot.get("target", 0.0))
        tolerance = float(snapshot.get("tolerance", DEFAULT_TOLERANCE))
        hold.tolerance = tolerance if tolerance > 0 else DEFAULT_TOLERANCE
        hold.drift = float(snapshot.get("drift", 0.0))
        hold.locked = bool(snapshot.get("locked", False))
        hold.corrections = int(snapshot.get("corrections", 0))
        hold.peak_drift = float(snapshot.get("peak_drift", 0.0))
        hold.sequence = int(snapshot.get("sequence", 0))
        hold.events = [HoldEvent.from_dict(item) for item in snapshot.get("events", [])]
        _trim(hold)
