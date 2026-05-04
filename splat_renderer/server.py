from __future__ import annotations

import base64
import time
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from renderer_core import CameraSpec, Pose2D, SplatRenderer
from scoring import ScoreMetric, image_error, image_file_to_tensor, rgb_to_gray, tensor_to_png_bytes


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


class ScoreBatchRequest(BaseModel):
    poses: list[PoseModel]
    camera: CameraModel
    observation_png_base64: str
    metric: ScoreMetric = "hybrid"
    ssim_window_size: int = 11
    hybrid_ssim_weight: float = 0.50
    hybrid_l1_weight: float = 0.25
    hybrid_gradient_weight: float = 0.25
    packed: bool | None = None
    radius_clip: float | None = None
    sh_degree: int | None = None
    max_batch_size: int | None = None


class ScoreBatchResponse(BaseModel):
    scores: list[float]
    best_index: int
    best_render_png_base64: str
    elapsed_ms: float


renderer: SplatRenderer | None = None


def to_pose(pose: PoseModel) -> Pose2D:
    return Pose2D(x=pose.x, y=pose.y, yaw=pose.yaw)


def to_camera(camera: CameraModel) -> CameraSpec:
    return CameraSpec(
        width=camera.width,
        height=camera.height,
        fx=camera.fx,
        fy=camera.fy,
        cx=camera.cx,
        cy=camera.cy,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    global renderer
    renderer = SplatRenderer(
        ply_path=os.environ.get("SPLAT_PATH", "/workspace/splat.ply"),
    )
    yield
    renderer = None


app = FastAPI(title="Splat Renderer", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "renderer_loaded": renderer is not None,
        "splat_path": str(renderer.ply_path) if renderer is not None else None,
        "gaussians": int(renderer.means.shape[0]) if renderer is not None else None,
    }


@app.post("/render", response_class=Response)
def render(request: RenderRequest) -> Response:
    if renderer is None:
        raise HTTPException(status_code=503, detail="Renderer not initialized")
    rgb = renderer.render_batch(
        poses=[to_pose(request.pose)],
        camera=to_camera(request.camera),
        packed=request.packed,
        white_background=request.white_background,
        radius_clip=request.radius_clip,
        sh_degree=request.sh_degree,
    )[0]
    return Response(content=tensor_to_png_bytes(rgb), media_type="image/png")


@app.post("/render_batch")
def render_batch(request: RenderBatchRequest) -> dict:
    if renderer is None:
        raise HTTPException(status_code=503, detail="Renderer not initialized")
    start = time.perf_counter()
    rgb_batch = renderer.render_batch_chunked(
        poses=[to_pose(pose) for pose in request.poses],
        camera=to_camera(request.camera),
        packed=request.packed,
        white_background=request.white_background,
        radius_clip=request.radius_clip,
        sh_degree=request.sh_degree,
    )
    images = [base64.b64encode(tensor_to_png_bytes(rgb_batch[index])).decode("ascii") for index in range(rgb_batch.shape[0])]
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {"images_png_base64": images, "count": len(images), "elapsed_ms": elapsed_ms, "shape": list(rgb_batch.shape)}


@app.post("/score_batch", response_model=ScoreBatchResponse)
def score_batch(request: ScoreBatchRequest) -> ScoreBatchResponse:
    if renderer is None:
        raise HTTPException(status_code=503, detail="Renderer not initialized")
    start = time.perf_counter()
    try:
        observation_image = base64.b64decode(request.observation_png_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid observation_png_base64: {exc}") from exc

    camera = to_camera(request.camera)
    render_rgb = renderer.render_batch_chunked(
        poses=[to_pose(pose) for pose in request.poses],
        camera=camera,
        packed=request.packed,
        radius_clip=request.radius_clip,
        sh_degree=request.sh_degree,
        max_batch_size=request.max_batch_size,
    )
    obs_rgb = image_file_to_tensor(observation_image, camera.width, camera.height, renderer.device)
    render_gray = rgb_to_gray(render_rgb)
    obs_gray = rgb_to_gray(obs_rgb)
    errors = image_error(
        render_rgb,
        obs_rgb,
        render_gray,
        obs_gray,
        request.metric,
        request.ssim_window_size,
        request.hybrid_ssim_weight,
        request.hybrid_l1_weight,
        request.hybrid_gradient_weight,
    )
    best_index = int(errors.argmin().item())
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return ScoreBatchResponse(
        scores=errors.detach().cpu().tolist(),
        best_index=best_index,
        best_render_png_base64=base64.b64encode(tensor_to_png_bytes(render_rgb[best_index])).decode("ascii"),
        elapsed_ms=elapsed_ms,
    )
