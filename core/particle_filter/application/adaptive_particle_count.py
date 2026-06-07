"""Adaptive particle-count controller for renderer-backed localization."""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.config.models import AdaptiveParticleCountSettings
from core.particle_filter.domain.particle import Particle
from core.particle_filter.domain.pose import wrap_angle


@dataclass(frozen=True)
class AdaptiveParticleCountStatus:
    enabled: bool
    min_particle_count: int
    medium_particle_count: int
    max_particle_count: int
    target_particle_count: int
    stable_required_updates: int
    unstable_required_updates: int
    xy_spread_stable_meters: float
    xy_spread_unstable_meters: float
    yaw_spread_stable_radians: float
    yaw_spread_unstable_radians: float
    best_score_stable_threshold: float
    median_score_stable_threshold: float
    stable_update_count: int
    unstable_update_count: int
    xy_spread_meters: float
    yaw_spread_radians: float
    last_resize_reason: str


@dataclass(frozen=True)
class AdaptiveParticleCountDecision:
    target_particle_count: int
    resize_reason: str | None = None


class AdaptiveParticleCountController:
    """Chooses a smaller or larger particle count from spread and renderer-score signals."""

    def __init__(self, settings: AdaptiveParticleCountSettings, *, configured_particle_count: int) -> None:
        self.settings = settings
        self._configured_particle_count = int(configured_particle_count)
        self._validate_resolved_counts()
        self._target_particle_count = self.max_particle_count
        self._stable_update_count = 0
        self._unstable_update_count = 0
        self._last_resize_reason = ""
        self._last_xy_spread_meters = 0.0
        self._last_yaw_spread_radians = 0.0

    @property
    def max_particle_count(self) -> int:
        return int(self.settings.max_particle_count or self._configured_particle_count)

    def reset(self) -> AdaptiveParticleCountDecision:
        self._stable_update_count = 0
        self._unstable_update_count = 0
        self._target_particle_count = self.max_particle_count
        self._last_resize_reason = "reset"
        return AdaptiveParticleCountDecision(target_particle_count=self._target_particle_count, resize_reason="reset")

    def set_configured_particle_count(self, particle_count: int) -> None:
        self._configured_particle_count = max(int(particle_count), self.settings.min_particle_count)
        if self.settings.max_particle_count is None:
            self._target_particle_count = min(self._target_particle_count, self.max_particle_count)

    def update_settings(self, settings: AdaptiveParticleCountSettings) -> None:
        self.settings = settings
        self._validate_resolved_counts()
        self._target_particle_count = min(max(self._target_particle_count, settings.min_particle_count), self.max_particle_count)

    def update(
        self,
        *,
        particles: list[Particle],
        current_particle_count: int,
        best_score: float,
        median_score: float,
        random_particle_count: int,
    ) -> AdaptiveParticleCountDecision:
        if not self.settings.enabled:
            return AdaptiveParticleCountDecision(target_particle_count=current_particle_count)

        self._last_xy_spread_meters, self._last_yaw_spread_radians = self._weighted_spreads(particles)
        is_stable = (
            self._last_xy_spread_meters <= self.settings.xy_spread_stable_meters
            and self._last_yaw_spread_radians <= self.settings.yaw_spread_stable_radians
            and best_score <= self.settings.best_score_stable_threshold
            and median_score <= self.settings.median_score_stable_threshold
            and random_particle_count <= 0
        )
        is_unstable = (
            self._last_xy_spread_meters >= self.settings.xy_spread_unstable_meters
            or self._last_yaw_spread_radians >= self.settings.yaw_spread_unstable_radians
            or best_score > self.settings.best_score_stable_threshold
            or median_score > self.settings.median_score_stable_threshold
            or random_particle_count > 0
        )

        self._stable_update_count = self._stable_update_count + 1 if is_stable else 0
        self._unstable_update_count = self._unstable_update_count + 1 if is_unstable else 0

        next_target = self._target_particle_count
        resize_reason = None
        if self._unstable_update_count >= self.settings.unstable_required_updates:
            next_target = self.max_particle_count
            resize_reason = "unstable"
        elif self._stable_update_count >= self.settings.stable_required_updates:
            next_target = self._next_lower_target(current_particle_count)
            if next_target < current_particle_count:
                resize_reason = "stable"

        next_target = self._clamp_count(next_target)
        if next_target != current_particle_count:
            self._target_particle_count = next_target
            self._last_resize_reason = resize_reason or "target"
            self._stable_update_count = 0
            self._unstable_update_count = 0
            return AdaptiveParticleCountDecision(target_particle_count=next_target, resize_reason=self._last_resize_reason)

        self._target_particle_count = current_particle_count
        return AdaptiveParticleCountDecision(target_particle_count=current_particle_count)

    def status(self) -> AdaptiveParticleCountStatus:
        return AdaptiveParticleCountStatus(
            enabled=self.settings.enabled,
            min_particle_count=self.settings.min_particle_count,
            medium_particle_count=self.settings.medium_particle_count,
            max_particle_count=self.max_particle_count,
            target_particle_count=self._target_particle_count,
            stable_required_updates=self.settings.stable_required_updates,
            unstable_required_updates=self.settings.unstable_required_updates,
            xy_spread_stable_meters=self.settings.xy_spread_stable_meters,
            xy_spread_unstable_meters=self.settings.xy_spread_unstable_meters,
            yaw_spread_stable_radians=self.settings.yaw_spread_stable_radians,
            yaw_spread_unstable_radians=self.settings.yaw_spread_unstable_radians,
            best_score_stable_threshold=self.settings.best_score_stable_threshold,
            median_score_stable_threshold=self.settings.median_score_stable_threshold,
            stable_update_count=self._stable_update_count,
            unstable_update_count=self._unstable_update_count,
            xy_spread_meters=self._last_xy_spread_meters,
            yaw_spread_radians=self._last_yaw_spread_radians,
            last_resize_reason=self._last_resize_reason,
        )

    def _next_lower_target(self, current_particle_count: int) -> int:
        if current_particle_count > self.settings.medium_particle_count:
            return self.settings.medium_particle_count
        return self.settings.min_particle_count

    def _clamp_count(self, particle_count: int) -> int:
        return min(self.max_particle_count, max(self.settings.min_particle_count, int(particle_count)))

    def _validate_resolved_counts(self) -> None:
        if self.max_particle_count < self.settings.min_particle_count:
            raise ValueError("adaptive_particle_count resolved max_particle_count must be >= min_particle_count")
        if self.settings.medium_particle_count < self.settings.min_particle_count:
            raise ValueError("adaptive_particle_count.medium_particle_count must be >= min_particle_count")

    @staticmethod
    def _weighted_spreads(particles: list[Particle]) -> tuple[float, float]:
        if not particles:
            return 0.0, 0.0

        total_weight = sum(max(0.0, particle.weight) for particle in particles)
        if total_weight <= 0.0:
            return 0.0, 0.0

        mean_x = sum(particle.pose.x * max(0.0, particle.weight) for particle in particles) / total_weight
        mean_y = sum(particle.pose.y * max(0.0, particle.weight) for particle in particles) / total_weight
        sine_sum = sum(math.sin(particle.pose.yaw) * max(0.0, particle.weight) for particle in particles)
        cosine_sum = sum(math.cos(particle.pose.yaw) * max(0.0, particle.weight) for particle in particles)
        mean_yaw = math.atan2(sine_sum, cosine_sum)

        xy_variance = 0.0
        yaw_variance = 0.0
        for particle in particles:
            weight = max(0.0, particle.weight) / total_weight
            dx = particle.pose.x - mean_x
            dy = particle.pose.y - mean_y
            dyaw = wrap_angle(particle.pose.yaw - mean_yaw)
            xy_variance += (dx * dx + dy * dy) * weight
            yaw_variance += (dyaw * dyaw) * weight
        return math.sqrt(max(0.0, xy_variance)), math.sqrt(max(0.0, yaw_variance))
