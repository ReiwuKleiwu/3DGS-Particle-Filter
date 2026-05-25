"""Adaptive random-particle recovery tracking for augmented MCL."""

from __future__ import annotations

from dataclasses import dataclass

from core.config.models import RecoverySettings


@dataclass
class AugmentedMclRecoveryTracker:
    settings: RecoverySettings
    w_slow: float | None = None
    w_fast: float | None = None

    def reset(self) -> None:
        self.w_slow = None
        self.w_fast = None

    def update(self, measurement_likelihood: float) -> float:
        if not self.settings.enabled:
            return 0.0

        likelihood = max(float(measurement_likelihood), 1e-300)
        if self.w_slow is None or self.w_fast is None:
            self.w_slow = likelihood
            self.w_fast = likelihood
            return self._clamp_ratio(self.settings.random_particle_floor_ratio)

        self.w_slow = self.w_slow + self.settings.alpha_slow * (likelihood - self.w_slow)
        self.w_fast = self.w_fast + self.settings.alpha_fast * (likelihood - self.w_fast)
        if self.w_slow <= 0.0:
            return self._clamp_ratio(self.settings.random_particle_floor_ratio)

        ratio = 1.0 - (self.w_fast / self.w_slow)
        return self._clamp_ratio(ratio)

    def _clamp_ratio(self, ratio: float) -> float:
        lower = max(0.0, self.settings.random_particle_floor_ratio)
        upper = min(1.0, max(lower, self.settings.random_particle_max_ratio))
        return min(upper, max(lower, ratio))
