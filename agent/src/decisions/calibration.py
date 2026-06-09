"""Calibration service — auto-tunes debate thresholds from reflections.

Applies parameter-change recommendations produced by
:class:`ReflectionService` to the live :class:`DebateConfig`, persists
each adjustment, and supports reverting individual changes.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime

from src.debate.schemas import DebateConfig
from src.decisions.reflection import ReflectionReport
from src.decisions.store import DecisionStore


@dataclass(frozen=True)
class CalibrationUpdate:
    """Record of a single parameter adjustment."""

    id: int
    reflection_id: int
    parameter: str
    old_value: float
    new_value: float
    reason: str
    applied_at: str
    reverted_at: str | None = None


class CalibrationService:
    """Applies reflection recommendations to DebateConfig thresholds."""

    CALIBRATABLE_PARAMS: set[str] = {
        "confidence_skip_below",
        "confidence_log_below",
        "confidence_trade_above",
    }

    def __init__(
        self,
        store: DecisionStore,
        base_config: DebateConfig,
    ) -> None:
        self._store = store
        self._base_config = base_config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_reflection(
        self,
        report: ReflectionReport,
        reflection_id: int,
    ) -> list[CalibrationUpdate]:
        """Apply calibratable recommendations and return the updates."""
        current_config = self.get_current_config()
        updates: list[CalibrationUpdate] = []

        for rec in report.recommendations:
            param = rec.get("parameter", "")
            if param not in self.CALIBRATABLE_PARAMS:
                continue

            old_value = float(getattr(current_config, param))
            new_value = float(rec["recommended_value"])
            reason = rec.get("reason", "")

            row_id = self._store.save_calibration_update({
                "reflection_id": reflection_id,
                "parameter": param,
                "old_value": old_value,
                "new_value": new_value,
                "reason": reason,
            })

            # Fetch the saved row to get applied_at timestamp
            active = self._store.get_active_calibrations()
            saved = next((c for c in active if c["id"] == row_id), None)
            applied_at = saved["applied_at"] if saved else datetime.now().isoformat()

            updates.append(CalibrationUpdate(
                id=row_id,
                reflection_id=reflection_id,
                parameter=param,
                old_value=old_value,
                new_value=new_value,
                reason=reason,
                applied_at=applied_at,
            ))

        return updates

    def get_current_config(self) -> DebateConfig:
        """Return a DebateConfig with all active calibrations applied.

        Starts from ``base_config`` and overlays every non-reverted
        calibration update (oldest first so later updates win).
        """
        active = self._store.get_active_calibrations()

        if not active:
            return self._base_config

        # Build overrides dict — active calibrations are returned newest
        # first, so we reverse to apply oldest first (latest wins).
        overrides: dict[str, float] = {}
        for cal in reversed(active):
            param = cal["parameter"]
            if param in self.CALIBRATABLE_PARAMS:
                overrides[param] = cal["new_value"]

        # Create new config with overrides applied
        base_dict = dataclasses.asdict(self._base_config)
        for param, value in overrides.items():
            # DebateConfig uses int for thresholds
            base_dict[param] = int(value)

        return DebateConfig(**base_dict)

    def revert_update(self, calibration_id: int) -> None:
        """Mark a calibration update as reverted."""
        self._store.revert_calibration(calibration_id)
