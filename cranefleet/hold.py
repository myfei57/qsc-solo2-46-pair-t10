"""Position hold: keep the slewed boom on target against wind drift.

Once the slew is in position the operator engages the hold and the module
watches the boom angle. Drift inside the tolerance is only tracked, while
drift past the tolerance is corrected back to the target angle, raises an
alert and locks the slew until someone on site confirms the correction.
Every drift excursion, correction and confirmation is recorded in a bounded
event log (and mirrored into the audit trail) so the hold period can be
reviewed later; the state survives a restart through the store.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from . import slew as slew_module
from .audit import Audit
from .errors import (
    HoldAlreadyEngaged,
    HoldLocked,
    HoldNotEngaged,
    HoldNotLocked,
    HoldSlewMoving,
    StoreNotFound,
)
from .slew import Slew
from .store import Store

STATE_INACTIVE = "inactive"
STATE_HOLDING = "holding"
STATE_LOCKED = "locked"

EVENT_ENGAGE = "hold-engage"
EVENT_DRIFT = "hold-drift"
EVENT_CORRECT = "hold-correct"
EVENT_CONFIRM = "hold-confirm"
EVENT_RELEASE = "hold-release"

EVENT_CAPACITY = 200
DEFAULT_TOLERANCE = 0.5


@dataclass(frozen=True)
class HoldEvent:
    """One recorded hold event: engagement, drift excursion, correction,

    on-site confirmation or release."""

    event_id: str
    recorded_at: str
    kind: str
    target: float
    angle: float
    drift: float
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.event_id,
            "at": self.recorded_at,
            "kind": self.kind,
            "target": self.target,
            "angle": self.angle,
            "drift": self.drift,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "HoldEvent":
        return cls(
            event_id=str(payload["id"]),
            recorded_at=str(payload["at"]),
            kind=str(payload["kind"]),
            target=float(payload["target"]),
            angle=float(payload["angle"]),
            drift=float(payload["drift"]),
            detail=str(payload.get("detail", "")),
        )


@dataclass(frozen=True)
class DriftResult:
    """The outcome of applying one wind-driven offset."""

    angle: float
    deviation: float
    corrected: bool
    locked: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "angle": self.angle,
            "deviation": self.deviation,
            "corrected": self.corrected,
            "locked": self.locked,
        }


class Hold:
    """Mutable position-hold state of one crane's slewing gear."""

    def __init__(self, crane_id: str) -> None:
        self._lock = threading.Lock()
        self.crane_id = crane_id
        self.state = STATE_INACTIVE
        self.target = 0.0
        self.tolerance = DEFAULT_TOLERANCE
        self.alert = False
        self.last_drift = 0.0
        self.peak_drift = 0.0
        self.drift_observations = 0
        self.correction_count = 0
        self.confirmed_by = ""
        self.events: list[HoldEvent] = []

    def state_now(self) -> str:
        with self._lock:
            return self.state

    def target_now(self) -> float:
        with self._lock:
            return self.target

    def tolerance_now(self) -> float:
        with self._lock:
            return self.tolerance

    def has_alert(self) -> bool:
        with self._lock:
            return self.alert

    def last_drift_now(self) -> float:
        with self._lock:
            return self.last_drift

    def peak_drift_now(self) -> float:
        with self._lock:
            return self.peak_drift

    def drift_observations_now(self) -> int:
        with self._lock:
            return self.drift_observations

    def correction_count_now(self) -> int:
        with self._lock:
            return self.correction_count

    def confirmed_by_now(self) -> str:
        with self._lock:
            return self.confirmed_by

    def event_count(self) -> int:
        with self._lock:
            return len(self.events)


def _append(hold: Hold, kind: str, angle: float, drift: float, detail: str) -> HoldEvent:
    """Append one event to the bounded log; the caller holds ``hold._lock``."""

    event = HoldEvent(
        event_id=uuid.uuid4().hex,
        recorded_at=datetime.now(timezone.utc).isoformat(),
        kind=kind,
        target=hold.target,
        angle=angle,
        drift=drift,
        detail=detail,
    )
    hold.events.append(event)
    if len(hold.events) > EVENT_CAPACITY:
        del hold.events[: len(hold.events) - EVENT_CAPACITY]
    return event


def engage(
    hold: Hold,
    slew: Slew,
    tolerance: float,
    recorder: Audit | None = None,
    target: float | None = None,
) -> float:
    """Arm the hold once the slew is in position; returns the held angle."""

    if slew.is_moving():
        raise HoldSlewMoving("slew is still moving")
    with hold._lock:  # noqa: SLF001 - module-level functions share hold state
        if hold.state != STATE_INACTIVE:
            raise HoldAlreadyEngaged("hold already engaged")
        hold.target = slew.angle_now() if target is None else target
        hold.tolerance = tolerance if tolerance > 0 else DEFAULT_TOLERANCE
        hold.state = STATE_HOLDING
        hold.alert = False
        hold.last_drift = 0.0
        hold.peak_drift = 0.0
        hold.drift_observations = 0
        hold.correction_count = 0
        hold.confirmed_by = ""
        held_angle = hold.target
        _append(hold, EVENT_ENGAGE, held_angle, 0.0, "hold engaged")
    slew_module.set_hold_lock(slew, True)
    if recorder is not None:
        recorder.record(hold.crane_id, EVENT_ENGAGE, "target=" + slew_module.format_angle(held_angle))
    return held_angle


def drift(hold: Hold, slew: Slew, offset: float, recorder: Audit | None = None) -> DriftResult:
    """Apply one wind-driven offset and correct back when out of tolerance.

    A correction locks the slew and raises the alert until :func:`confirm`
    acknowledges it on site.
    """

    with hold._lock:  # noqa: SLF001
        if hold.state == STATE_LOCKED:
            raise HoldLocked("correction pending on-site confirmation")
        if hold.state != STATE_HOLDING:
            raise HoldNotEngaged("hold not engaged")
        target = hold.target
        tolerance = hold.tolerance
    angle = slew_module.apply_drift(slew, offset)
    deviation = angle - target
    with hold._lock:  # noqa: SLF001
        hold.drift_observations += 1
        hold.last_drift = deviation
        if abs(deviation) > abs(hold.peak_drift):
            hold.peak_drift = deviation
        if abs(deviation) <= tolerance:
            return DriftResult(angle=angle, deviation=deviation, corrected=False, locked=False)
        _append(hold, EVENT_DRIFT, angle, deviation, "drift past tolerance")
        hold.state = STATE_LOCKED
        hold.alert = True
        hold.correction_count += 1
        correction = hold.correction_count
    slew_module.restore_angle(slew, target)
    with hold._lock:  # noqa: SLF001
        _append(hold, EVENT_CORRECT, target, deviation, f"corrected to target #{correction}")
    if recorder is not None:
        recorder.record(
            hold.crane_id,
            EVENT_DRIFT,
            f"drift={deviation:.2f} tolerance={tolerance:.2f}",
        )
        recorder.record(hold.crane_id, EVENT_CORRECT, "target=" + slew_module.format_angle(target))
    return DriftResult(angle=target, deviation=deviation, corrected=True, locked=True)


def confirm(hold: Hold, operator: str = "", recorder: Audit | None = None) -> None:
    """Acknowledge the pending correction after the on-site check."""

    with hold._lock:  # noqa: SLF001
        if hold.state != STATE_LOCKED:
            raise HoldNotLocked("no correction awaiting confirmation")
        hold.state = STATE_HOLDING
        hold.alert = False
        hold.confirmed_by = operator
        _append(hold, EVENT_CONFIRM, hold.target, hold.last_drift, "confirmed by " + (operator or "site"))
    if recorder is not None:
        recorder.record(hold.crane_id, EVENT_CONFIRM, "by=" + (operator or "site"))


def release(hold: Hold, slew: Slew, recorder: Audit | None = None) -> None:
    """Release the hold so the slew can move again.

    A locked hold must be confirmed on site before it can be released.
    """

    with hold._lock:  # noqa: SLF001
        if hold.state == STATE_LOCKED:
            raise HoldLocked("correction pending on-site confirmation")
        if hold.state != STATE_HOLDING:
            raise HoldNotEngaged("hold not engaged")
        hold.state = STATE_INACTIVE
        _append(hold, EVENT_RELEASE, hold.target, hold.last_drift, "hold released")
    slew_module.set_hold_lock(slew, False)
    if recorder is not None:
        recorder.record(hold.crane_id, EVENT_RELEASE, "hold released")


def is_active(hold: Hold) -> bool:
    """Report whether the hold currently keeps the slew locked."""

    return hold.state_now() in (STATE_HOLDING, STATE_LOCKED)


def is_locked(hold: Hold) -> bool:
    """Report whether a correction is waiting for on-site confirmation."""

    return hold.state_now() == STATE_LOCKED


def events_snapshot(hold: Hold, limit: int = 50) -> list[HoldEvent]:
    """Return the newest hold events first."""

    with hold._lock:  # noqa: SLF001
        events = list(hold.events)
    if limit <= 0 or limit > len(events):
        limit = len(events)
    return list(reversed(events))[:limit]


def _hold_key(crane_id: str) -> str:
    return "hold/" + crane_id


def persist(hold: Hold, store: Store) -> None:
    with hold._lock:  # noqa: SLF001
        payload = json.dumps(
            {
                "state": hold.state,
                "target": hold.target,
                "tolerance": hold.tolerance,
                "alert": hold.alert,
                "last_drift": hold.last_drift,
                "peak_drift": hold.peak_drift,
                "drift_observations": hold.drift_observations,
                "correction_count": hold.correction_count,
                "confirmed_by": hold.confirmed_by,
                "events": [event.to_dict() for event in hold.events],
            }
        ).encode("utf-8")
    store.save(_hold_key(hold.crane_id), payload)


def restore(hold: Hold, store: Store) -> None:
    try:
        payload = store.load(_hold_key(hold.crane_id))
    except StoreNotFound:
        return
    snapshot = json.loads(payload.decode("utf-8"))
    with hold._lock:  # noqa: SLF001
        hold.state = snapshot["state"]
        hold.target = snapshot["target"]
        hold.tolerance = snapshot["tolerance"]
        hold.alert = snapshot["alert"]
        hold.last_drift = snapshot["last_drift"]
        hold.peak_drift = snapshot["peak_drift"]
        hold.drift_observations = snapshot["drift_observations"]
        hold.correction_count = snapshot["correction_count"]
        hold.confirmed_by = snapshot["confirmed_by"]
        hold.events = [HoldEvent.from_dict(item) for item in snapshot["events"]]
