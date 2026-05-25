"""Samples robot poses uniformly from the traversable cells of a ROS occupancy map."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from core.particle_filter.domain.pose import Pose2D


@dataclass(frozen=True)
class FreeSpacePoseSampler:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float
    free_cells: np.ndarray
    global_yaw_uniform: bool = True

    @classmethod
    def from_map_yaml(cls, map_yaml_path: str | Path, *, global_yaw_uniform: bool = True) -> "FreeSpacePoseSampler":
        resolved_map_yaml_path = Path(map_yaml_path).resolve()
        with resolved_map_yaml_path.open("r", encoding="utf-8") as map_file:
            raw_metadata = yaml.safe_load(map_file) or {}

        image_path = (resolved_map_yaml_path.parent / raw_metadata["image"]).resolve()
        with Image.open(image_path) as image:
            grayscale = np.asarray(image.convert("L"), dtype=np.float32)

        negate = int(raw_metadata.get("negate", 0))
        if negate:
            occupancy_probability = grayscale / 255.0
        else:
            occupancy_probability = (255.0 - grayscale) / 255.0

        free_threshold = float(raw_metadata.get("free_thresh", 0.196))
        free_mask = occupancy_probability < free_threshold
        free_cells = np.argwhere(free_mask)
        if free_cells.size == 0:
            raise ValueError(f"No traversable cells found in map {resolved_map_yaml_path}")

        origin = raw_metadata.get("origin", [-10.0, -10.0, 0.0])
        return cls(
            width=int(grayscale.shape[1]),
            height=int(grayscale.shape[0]),
            resolution=float(raw_metadata.get("resolution", 0.05)),
            origin_x=float(origin[0]),
            origin_y=float(origin[1]),
            origin_yaw=float(origin[2]),
            free_cells=free_cells,
            global_yaw_uniform=global_yaw_uniform,
        )

    def sample_pose(self, rng: random.Random | None = None) -> Pose2D:
        rng = rng or random.Random()
        cell_index = rng.randrange(len(self.free_cells))
        row, col = self.free_cells[cell_index]
        image_x = float(col) + rng.random()
        image_y = float(row) + rng.random()
        map_y = self.height - image_y

        cosine = np.cos(self.origin_yaw)
        sine = np.sin(self.origin_yaw)
        dx = self.resolution * (cosine * image_x - sine * map_y)
        dy = self.resolution * (sine * image_x + cosine * map_y)
        yaw = rng.uniform(-np.pi, np.pi) if self.global_yaw_uniform else self.origin_yaw
        return Pose2D(
            x=self.origin_x + float(dx),
            y=self.origin_y + float(dy),
            yaw=float(yaw),
        )
