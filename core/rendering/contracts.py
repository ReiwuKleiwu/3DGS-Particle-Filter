"""Pydantic request and response models shared across rendering API endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field

from core.rendering.scoring import ScoreMetric


class CameraModel(BaseModel):
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    fx: float
    fy: float
    cx: float
    cy: float


class PoseModel(BaseModel):
    x: float
    y: float
    yaw: float


class RenderRequest(BaseModel):
    pose: PoseModel
    camera: CameraModel
    packed: bool | None = None
    white_background: bool = False
    radius_clip: float | None = None
    sh_degree: int | None = None


class RenderBatchRequest(BaseModel):
    poses: list[PoseModel]
    camera: CameraModel
    packed: bool | None = None
    white_background: bool = False
    radius_clip: float | None = None
    sh_degree: int | None = None
    max_batch_size: int | None = None


class ScoreBatchRequest(BaseModel):
    poses: list[PoseModel]
    camera: CameraModel
    observation_png_base64: str
    include_best_render_preview: bool = True
    metric: ScoreMetric = "hybrid"
    ssim_window_size: int = 11
    hybrid_ssim_weight: float = 0.50
    hybrid_l1_weight: float = 0.25
    hybrid_gradient_weight: float = 0.25
    lpips_top_k: int = 0
    lpips_weight: float = 0.0
    lpips_net: str = "alex"
    packed: bool | None = None
    radius_clip: float | None = None
    sh_degree: int | None = None
    max_batch_size: int | None = None


class ScoreBatchResponse(BaseModel):
    scores: list[float]
    best_index: int
    best_render_png_base64: str
    elapsed_ms: float
    diagnostics: dict[str, float | int | str | bool] | None = None
