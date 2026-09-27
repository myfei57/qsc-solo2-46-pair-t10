"""Domain failures raised by the control service.

Every failure carries a stable type so the HTTP layer can map it to a status
code without inspecting message text.
"""

from __future__ import annotations


class CraneFleetError(Exception):
    """Base class for every failure raised by the control service."""


class StoreError(CraneFleetError):
    """Raised when the persistence layer cannot satisfy a request."""


class StoreNotFound(StoreError):
    """Raised when a store key has never been written."""


class ZoneError(CraneFleetError):
    """Raised when a collision-zone operation is rejected."""


class ZoneNotDurable(ZoneError):
    """Raised when a swing is attempted before the zone state is persisted."""


class ClusterError(CraneFleetError):
    """Raised when a fleet-level operation is rejected."""


class NoCranesSurveyed(ClusterError):
    """Raised when a baseline refresh runs before any crane is surveyed."""


class SlewError(CraneFleetError):
    """Base class for slewing failures."""


class SlewBlocked(SlewError):
    """Raised when an interlock (stop, wind or latch) holds the slew."""


class SlewBusy(SlewError):
    """Raised when a swing starts while the slew is already moving."""


class SlewIdle(SlewError):
    """Raised when a swing is applied while the slew is not moving."""


class HoldError(CraneFleetError):
    """Base class for position-hold failures."""


class HoldNotEngaged(HoldError):
    """Raised when a hold operation arrives while the hold is inactive."""


class HoldAlreadyEngaged(HoldError):
    """Raised when the hold is engaged again without a release."""


class HoldLocked(HoldError):
    """Raised while a correction still waits for on-site confirmation."""


class HoldNotLocked(HoldError):
    """Raised when a confirmation arrives without a pending correction."""


class HoldSlewMoving(HoldError):
    """Raised when the hold is engaged before the slew has come to rest."""


class TrolleyError(CraneFleetError):
    """Base class for trolley failures."""


class HookNotLowered(TrolleyError):
    """Raised when the trolley moves before the hook is lowered."""


class HoistError(CraneFleetError):
    """Base class for hoist failures."""


class HoistBusy(HoistError):
    """Raised when the drum is already turning."""


class FeedbackNotConfirmed(HoistError):
    """Raised when the brake is released before the speed feedback arrives."""


class WindError(CraneFleetError):
    """Base class for wind failures."""


class SlewStopNotAcknowledged(WindError):
    """Raised when the slew stop is not acknowledged during a wind alarm."""


class LimitError(CraneFleetError):
    """Base class for limit failures."""


class StaleRelease(LimitError):
    """Raised when a stale generation tries to release the limit latch."""


class ConfigError(CraneFleetError):
    """Raised when a site configuration payload is invalid."""
