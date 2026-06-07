"""Create an MP4 video from particle snapshot PNG frames."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an MP4 video from particle snapshot PNG frames.")
    parser.add_argument("--frames-dir", type=Path, required=True, help="Directory containing frame_*.png files.")
    parser.add_argument("--output", type=Path, default=None, help="Output MP4 path.")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--crf", type=int, default=20, help="x264 CRF quality value; lower is higher quality.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames_dir = args.frames_dir.resolve()
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Missing frames directory: {frames_dir}")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if not list(frames_dir.glob("frame_*.png")):
        raise FileNotFoundError(f"No frame_*.png files found in {frames_dir}")

    output = args.output.resolve() if args.output else frames_dir.with_suffix(".mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_pattern = str(frames_dir / "frame_*.png")
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(args.fps),
        "-pattern_type",
        "glob",
        "-i",
        frame_pattern,
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(args.crf),
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(command, check=True)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
