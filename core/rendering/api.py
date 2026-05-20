"""FastAPI application that serves batch rendering and observation-scoring endpoints."""

from __future__ import annotations

import base64
import hashlib
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
import torch

from core.rendering.backends import RendererBackend, create_renderer_backend
from core.rendering.lpips import lpips_error
from core.rendering.contracts import (
    CameraModel,
    PoseModel,
    RenderBatchRequest,
    RenderRequest,
    ScoreBatchRequest,
    ScoreBatchResponse,
)
from core.rendering.types import CameraSpec, Pose2D
from core.rendering.scoring import image_error, image_file_to_tensor, rgb_to_gray, tensor_to_png_bytes


renderer_backend: RendererBackend | None = None


def _preview_fingerprint(preview_png_bytes: bytes) -> tuple[int, str]:
    if not preview_png_bytes:
        return 0, ""
    return len(preview_png_bytes), hashlib.sha1(preview_png_bytes).hexdigest()[:12]


def _apply_lpips_rerank(
    *,
    base_scores: torch.Tensor,
    poses: list[Pose2D],
    camera: CameraSpec,
    obs_rgb: torch.Tensor,
    request: ScoreBatchRequest,
    current_best_index: int,
    current_best_render_png_bytes: bytes,
    current_diagnostics: dict[str, float | int | str | bool],
    full_render_rgb: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, bytes, dict[str, float | int | str | bool]]:
    if renderer_backend is None:
        return base_scores, current_best_index, current_best_render_png_bytes, current_diagnostics
    if request.lpips_top_k <= 0 or request.lpips_weight <= 0.0 or not poses:
        return base_scores, current_best_index, current_best_render_png_bytes, current_diagnostics

    top_k = min(request.lpips_top_k, len(poses))
    rerank_start = time.perf_counter()
    topk_indices = torch.topk(base_scores, k=top_k, largest=False).indices
    topk_index_list = [int(index) for index in topk_indices.detach().cpu().tolist()]
    if full_render_rgb is not None:
        rerank_rgb = full_render_rgb[topk_indices]
        lpips_render_ms = 0.0
    else:
        rerank_poses = [poses[index] for index in topk_index_list]
        render_start = time.perf_counter()
        rerank_rgb = renderer_backend.render_batch(
            poses=rerank_poses,
            camera=camera,
            packed=request.packed,
            radius_clip=request.radius_clip,
            sh_degree=request.sh_degree,
            max_batch_size=request.max_batch_size,
        )
        lpips_render_ms = (time.perf_counter() - render_start) * 1000.0

    score_start = time.perf_counter()
    lpips_values = lpips_error(rerank_rgb, obs_rgb, device=renderer_backend.device, net=request.lpips_net)
    lpips_score_ms = (time.perf_counter() - score_start) * 1000.0

    adjusted_scores = base_scores.clone()
    adjusted_scores[topk_indices] = adjusted_scores[topk_indices] + request.lpips_weight * lpips_values
    best_index = int(adjusted_scores.argmin().item())

    best_render_png_bytes = current_best_render_png_bytes
    if request.include_best_render_preview:
        topk_slot_by_index = {index: slot for slot, index in enumerate(topk_index_list)}
        if best_index in topk_slot_by_index:
            best_render_png_bytes = tensor_to_png_bytes(rerank_rgb[topk_slot_by_index[best_index]])
        elif full_render_rgb is not None:
            best_render_png_bytes = tensor_to_png_bytes(full_render_rgb[best_index])

    current_diagnostics["lpips_top_k"] = top_k
    current_diagnostics["lpips_weight"] = request.lpips_weight
    current_diagnostics["lpips_render_ms"] = lpips_render_ms
    current_diagnostics["lpips_score_ms"] = lpips_score_ms
    current_diagnostics["lpips_rerank_ms"] = (time.perf_counter() - rerank_start) * 1000.0
    current_diagnostics["lpips_best_changed"] = best_index != current_best_index
    return adjusted_scores, best_index, best_render_png_bytes, current_diagnostics


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
    global renderer_backend
    renderer_backend = create_renderer_backend()
    yield
    if renderer_backend is not None and hasattr(renderer_backend, "close"):
        renderer_backend.close()
    renderer_backend = None


app = FastAPI(title="Renderer Service", version="0.2.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Returns renderer readiness and backend metadata for health checks and startup polling."""
    return {
        "status": "ok",
        "renderer_loaded": renderer_backend is not None,
        "backend": renderer_backend.backend_name if renderer_backend is not None else None,
        "splat_path": str(renderer_backend.splat_path) if renderer_backend is not None else None,
        "gaussians": renderer_backend.gaussian_count if renderer_backend is not None else None,
    }


@app.post("/render", response_class=Response)
def render(request: RenderRequest) -> Response:
    """Renders a single pose and returns the resulting PNG image bytes."""
    if renderer_backend is None:
        raise HTTPException(status_code=503, detail="Renderer not initialized")
    rgb = renderer_backend.render_batch(
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
    """Renders multiple poses in one request and returns each image as base64 PNG."""
    if renderer_backend is None:
        raise HTTPException(status_code=503, detail="Renderer not initialized")
    start = time.perf_counter()
    rgb_batch = renderer_backend.render_batch(
        poses=[to_pose(pose) for pose in request.poses],
        camera=to_camera(request.camera),
        packed=request.packed,
        white_background=request.white_background,
        radius_clip=request.radius_clip,
        sh_degree=request.sh_degree,
        max_batch_size=request.max_batch_size,
    )
    images = [base64.b64encode(tensor_to_png_bytes(rgb_batch[index])).decode("ascii") for index in range(rgb_batch.shape[0])]
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {"images_png_base64": images, "count": len(images), "elapsed_ms": elapsed_ms, "shape": list(rgb_batch.shape)}


@app.post("/score_batch", response_model=ScoreBatchResponse)
def score_batch(request: ScoreBatchRequest) -> ScoreBatchResponse:
    """Renders and scores a batch of poses against one observation image."""
    if renderer_backend is None:
        raise HTTPException(status_code=503, detail="Renderer not initialized")
    total_start = time.perf_counter()
    try:
        decode_start = time.perf_counter()
        observation_image = base64.b64decode(request.observation_png_base64)
        decode_b64_ms = (time.perf_counter() - decode_start) * 1000.0
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid observation_png_base64: {exc}") from exc

    camera = to_camera(request.camera)
    poses = [to_pose(pose) for pose in request.poses]
    native_score = renderer_backend.score_batch_native(
        poses=poses,
        camera=camera,
        observation_png_bytes=observation_image,
        metric=request.metric,
        lpips_net=request.lpips_net,
        include_best_render_preview=request.include_best_render_preview,
        ssim_window_size=request.ssim_window_size,
        hybrid_ssim_weight=request.hybrid_ssim_weight,
        hybrid_l1_weight=request.hybrid_l1_weight,
        hybrid_gradient_weight=request.hybrid_gradient_weight,
        packed=request.packed,
        radius_clip=request.radius_clip,
        sh_degree=request.sh_degree,
        max_batch_size=request.max_batch_size,
    )
    if native_score is not None:
        diagnostics: dict[str, float | int | str | bool]
        best_render_png_bytes = native_score["best_render_png_bytes"]
        scores_tensor = torch.tensor(native_score["scores"], dtype=torch.float32, device=renderer_backend.device)
        best_index = int(native_score["best_index"])
        backend_diagnostics = dict(native_score.get("diagnostics") or {})
        if request.lpips_top_k > 0 and request.lpips_weight > 0.0:
            obs_rgb = image_file_to_tensor(observation_image, camera.width, camera.height, renderer_backend.device)
            scores_tensor, best_index, best_render_png_bytes, backend_diagnostics = _apply_lpips_rerank(
                base_scores=scores_tensor,
                poses=poses,
                camera=camera,
                obs_rgb=obs_rgb,
                request=request,
                current_best_index=best_index,
                current_best_render_png_bytes=best_render_png_bytes,
                current_diagnostics=backend_diagnostics,
            )
        elapsed_ms = (time.perf_counter() - total_start) * 1000.0
        diagnostics = {
            "backend": renderer_backend.backend_name,
            "pose_count": len(poses),
            "observation_b64_decode_ms": decode_b64_ms,
            "render_call_ms": float(backend_diagnostics.get("request_roundtrip_ms", 0.0)),
            "scoring_ms": float(backend_diagnostics.get("score_gpu_ms", 0.0)),
            "best_png_encode_ms": float(backend_diagnostics.get("best_png_encode_ms", 0.0)),
            "total_ms": elapsed_ms,
        }
        preview_bytes_len, preview_hash = _preview_fingerprint(best_render_png_bytes)
        diagnostics["preview_bytes"] = preview_bytes_len
        diagnostics["preview_hash"] = preview_hash
        for key, value in backend_diagnostics.items():
            diagnostics[f"backend_{key}"] = value
        return ScoreBatchResponse(
            scores=scores_tensor.detach().cpu().tolist(),
            best_index=best_index,
            best_render_png_base64=(
                base64.b64encode(best_render_png_bytes).decode("ascii")
                if best_render_png_bytes
                else ""
            ),
            elapsed_ms=elapsed_ms,
            diagnostics=diagnostics,
        )

    render_start = time.perf_counter()
    render_rgb = renderer_backend.render_batch(
        poses=poses,
        camera=camera,
        packed=request.packed,
        radius_clip=request.radius_clip,
        sh_degree=request.sh_degree,
        max_batch_size=request.max_batch_size,
    )
    render_call_ms = (time.perf_counter() - render_start) * 1000.0
    obs_tensor_start = time.perf_counter()
    obs_rgb = image_file_to_tensor(observation_image, camera.width, camera.height, renderer_backend.device)
    obs_tensor_ms = (time.perf_counter() - obs_tensor_start) * 1000.0
    render_gray = None
    obs_gray = None
    gray_ms = 0.0
    if request.metric != "lpips":
        gray_start = time.perf_counter()
        render_gray = rgb_to_gray(render_rgb)
        obs_gray = rgb_to_gray(obs_rgb)
        gray_ms = (time.perf_counter() - gray_start) * 1000.0
    score_start = time.perf_counter()
    if request.metric == "lpips":
        errors = lpips_error(render_rgb, obs_rgb, device=renderer_backend.device, net=request.lpips_net)
    else:
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
    score_ms = (time.perf_counter() - score_start) * 1000.0
    best_index = int(errors.argmin().item())
    if request.metric != "lpips" and request.lpips_top_k > 0 and request.lpips_weight > 0.0:
        diagnostics_lpips: dict[str, float | int | str | bool] = {}
        errors, best_index, _, diagnostics_lpips = _apply_lpips_rerank(
            base_scores=errors,
            poses=poses,
            camera=camera,
            obs_rgb=obs_rgb,
            request=request,
            current_best_index=best_index,
            current_best_render_png_bytes=b"",
            current_diagnostics=diagnostics_lpips,
            full_render_rgb=render_rgb,
        )
    encode_best_png_ms = 0.0
    best_render_png_base64 = ""
    if request.include_best_render_preview:
        encode_start = time.perf_counter()
        best_render_png_bytes = tensor_to_png_bytes(render_rgb[best_index])
        best_render_png_base64 = base64.b64encode(best_render_png_bytes).decode("ascii")
        encode_best_png_ms = (time.perf_counter() - encode_start) * 1000.0
    elapsed_ms = (time.perf_counter() - total_start) * 1000.0
    diagnostics: dict[str, float | int | str | bool] = {
        "backend": renderer_backend.backend_name,
        "pose_count": len(poses),
        "observation_b64_decode_ms": decode_b64_ms,
        "render_call_ms": render_call_ms,
        "observation_tensor_ms": obs_tensor_ms,
        "grayscale_ms": gray_ms,
        "scoring_ms": score_ms,
        "best_png_encode_ms": encode_best_png_ms,
        "total_ms": elapsed_ms,
    }
    preview_bytes_len, preview_hash = _preview_fingerprint(base64.b64decode(best_render_png_base64) if best_render_png_base64 else b"")
    diagnostics["preview_bytes"] = preview_bytes_len
    diagnostics["preview_hash"] = preview_hash
    if request.metric != "lpips" and request.lpips_top_k > 0 and request.lpips_weight > 0.0:
        for key, value in diagnostics_lpips.items():
            diagnostics[key] = value
    backend_diagnostics = renderer_backend.get_last_render_diagnostics()
    if backend_diagnostics:
        for key, value in backend_diagnostics.items():
            diagnostics[f"backend_{key}"] = value
    return ScoreBatchResponse(
        scores=errors.detach().cpu().tolist(),
        best_index=best_index,
        best_render_png_base64=best_render_png_base64,
        elapsed_ms=elapsed_ms,
        diagnostics=diagnostics,
    )
