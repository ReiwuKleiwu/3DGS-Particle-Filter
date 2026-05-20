"""Formats concise console diagnostics for live localization updates."""

from __future__ import annotations

from core.particle_filter.application.step_engine import LocalizationStepResult
from core.particle_filter.infrastructure.ros.observation import TurtleBotObservation


class LocalizationDiagnosticsFormatter:
    """Builds a single-line status message for each live localization update."""

    def format(
        self,
        *,
        update_count: int,
        paused: bool,
        observation: TurtleBotObservation,
        step_result: LocalizationStepResult,
    ) -> str:
        """Formats one human-readable status line summarizing a live localization update."""
        status_line = (
            f"update {update_count:04d} | "
            f"estimate x={step_result.estimated_pose.x:.3f}, "
            f"y={step_result.estimated_pose.y:.3f}, "
            f"yaw={step_result.estimated_pose.yaw:.3f} | "
            f"best_index={step_result.score_result.best_index} | "
            f"render+score={step_result.score_result.elapsed_milliseconds:.1f} ms | "
            f"resampled={'yes' if step_result.resampled else 'no'} | "
            f"paused={'yes' if paused else 'no'}"
        )
        diagnostics = step_result.score_result.diagnostics or {}
        if diagnostics:
            render_call_ms = diagnostics.get("render_call_ms")
            scoring_ms = diagnostics.get("scoring_ms")
            http_ms = diagnostics.get("client_http_roundtrip_ms")
            backend_server_ms = diagnostics.get("backend_server_elapsed_ms")
            backend_name = diagnostics.get("backend")
            backend_render_wall_ms = diagnostics.get("backend_render_wall_ms")
            backend_render_submit_overhead_ms = diagnostics.get("backend_render_submit_overhead_ms")
            backend_score_sync_ms = diagnostics.get("backend_score_sync_ms")
            backend_obs_upload_ms = diagnostics.get("backend_observation_upload_ms")
            backend_obs_preprocess_ms = diagnostics.get("backend_observation_preprocess_ms")
            backend_best_copy_d2d_ms = diagnostics.get("backend_best_copy_d2d_ms")
            backend_best_preview_d2h_ms = diagnostics.get("backend_best_preview_d2h_ms")
            backend_worker_residual_ms = diagnostics.get("backend_worker_residual_ms")
            backend_payload_build_ms = diagnostics.get("backend_payload_build_ms")
            backend_request_roundtrip_ms = diagnostics.get("backend_request_roundtrip_ms")
            backend_output_decode_ms = diagnostics.get("backend_output_decode_ms")
            backend_device_upload_ms = diagnostics.get("backend_device_upload_ms")
            preview_bytes = diagnostics.get("preview_bytes")
            preview_hash = diagnostics.get("preview_hash")
            lpips_rerank_ms = diagnostics.get("lpips_rerank_ms")
            lpips_render_ms = diagnostics.get("lpips_render_ms")
            lpips_score_ms = diagnostics.get("lpips_score_ms")
            lpips_best_changed = diagnostics.get("lpips_best_changed")
            if render_call_ms is not None:
                status_line += f" | render={float(render_call_ms):.1f} ms"
            if scoring_ms is not None:
                status_line += f" | score={float(scoring_ms):.1f} ms"
            if backend_server_ms is not None:
                backend_label = f"{backend_name}_core" if isinstance(backend_name, str) and backend_name else "backend_core"
                status_line += f" | {backend_label}={float(backend_server_ms):.1f} ms"
            if backend_render_wall_ms is not None:
                status_line += f" | render_wall={float(backend_render_wall_ms):.1f} ms"
            if backend_render_submit_overhead_ms is not None:
                status_line += f" | submit_ovh={float(backend_render_submit_overhead_ms):.1f} ms"
            if backend_score_sync_ms is not None:
                status_line += f" | score_sync={float(backend_score_sync_ms):.1f} ms"
            if backend_obs_upload_ms is not None:
                status_line += f" | obs_up={float(backend_obs_upload_ms):.1f} ms"
            if backend_obs_preprocess_ms is not None:
                status_line += f" | obs_prep={float(backend_obs_preprocess_ms):.1f} ms"
            if backend_best_copy_d2d_ms is not None:
                status_line += f" | best_d2d={float(backend_best_copy_d2d_ms):.1f} ms"
            if backend_best_preview_d2h_ms is not None:
                status_line += f" | best_d2h={float(backend_best_preview_d2h_ms):.1f} ms"
            if backend_worker_residual_ms is not None:
                status_line += f" | worker_res={float(backend_worker_residual_ms):.1f} ms"
            if backend_payload_build_ms is not None:
                status_line += f" | pl_bld={float(backend_payload_build_ms):.1f} ms"
            if backend_request_roundtrip_ms is not None:
                status_line += f" | rndtrip={float(backend_request_roundtrip_ms):.1f} ms"
            if backend_output_decode_ms is not None:
                status_line += f" | decode={float(backend_output_decode_ms):.1f} ms"
            if backend_device_upload_ms is not None:
                status_line += f" | up={float(backend_device_upload_ms):.1f} ms"
            if preview_bytes is not None:
                status_line += f" | prv={int(preview_bytes)}B"
            if preview_hash:
                status_line += f" | prvh={preview_hash}"
            if lpips_rerank_ms is not None:
                status_line += f" | lpips={float(lpips_rerank_ms):.1f} ms"
            if lpips_render_ms is not None:
                status_line += f" | lpips_rnd={float(lpips_render_ms):.1f} ms"
            if lpips_score_ms is not None:
                status_line += f" | lpips_scr={float(lpips_score_ms):.1f} ms"
            if lpips_best_changed is not None:
                status_line += f" | lpips_best={'yes' if bool(lpips_best_changed) else 'no'}"
            if http_ms is not None:
                status_line += f" | http={float(http_ms):.1f} ms"

        if observation.map_pose is not None:
            pose_error = observation.pose_error_against(step_result.estimated_pose)
            status_line += (
                f" | eval dx={pose_error.x:.3f}, "
                f"dy={pose_error.y:.3f}, "
                f"dyaw={pose_error.yaw:.3f}"
            )
        return status_line
