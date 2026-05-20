"""Shared value types used across rendering backends and API layers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraSpec:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float
