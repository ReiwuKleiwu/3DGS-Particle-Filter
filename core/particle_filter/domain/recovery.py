"""Adaptive random-particle recovery tracking for augmented MCL."""

from __future__ import annotations

from dataclasses import dataclass

from core.config.models import RecoverySettings


@dataclass
class AugmentedMclRecoveryTracker:
    settings: RecoverySettings
    w_slow: float | None = None
    w_fast: float | None = None
    consecutive_bad_updates: int = 0

    def reset(self) -> None:
        self.w_slow = None
        self.w_fast = None
        self.consecutive_bad_updates = 0

    def update(
        self,
        measurement_likelihood: float,
        *,
        metric_name: str | None = None,
        best_score: float | None = None,
        median_score: float | None = None,
    ) -> float:
        if not self.settings.enabled:
            return 0.0
        if self.settings.strategy == "absolute_score":
            return self._update_absolute_score(
                metric_name=metric_name,
                best_score=best_score,
                median_score=median_score,
            )
        return self._update_augmented_mcl(measurement_likelihood)

    def _update_augmented_mcl(self, measurement_likelihood: float) -> float:
        self.consecutive_bad_updates = 0
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

    def _update_absolute_score(
        self,
        *,
        metric_name: str | None,
        best_score: float | None,
        median_score: float | None,
    ) -> float:
        if best_score is None or median_score is None:
            self.consecutive_bad_updates = 0
            return self._clamp_ratio(self.settings.random_particle_floor_ratio)

        profile = self._absolute_score_profile(metric_name)
        is_bad = (
            best_score > float(profile["best_score_threshold"])
            and median_score > float(profile["median_score_threshold"])
        )
        if is_bad:
            self.consecutive_bad_updates += 1
        else:
            self.consecutive_bad_updates = 0

        if self.consecutive_bad_updates < max(1, int(profile["consecutive_bad_updates"])):
            return self._clamp_ratio(self.settings.random_particle_floor_ratio)
        return self._clamp_ratio(float(profile["random_particle_ratio"]))

    def _absolute_score_profile(self, metric_name: str | None) -> dict[str, float | int]:
        profiles = self.settings.absolute_score_profiles
        metric_key = str(metric_name or "").strip().lower()
        return profiles.get(metric_key) or profiles["default"]

    def _clamp_ratio(self, ratio: float) -> float:
        lower = max(0.0, self.settings.random_particle_floor_ratio)
        upper = min(1.0, max(lower, self.settings.random_particle_max_ratio))
        return min(upper, max(lower, ratio))
