FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    glslang-tools \
    libglfw3-dev \
    libvulkan-dev \
    libxcursor-dev \
    libxi-dev \
    libxinerama-dev \
    libxrandr-dev \
    ninja-build \
    pkg-config \
    vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml
COPY core /app/core

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install .[rendering]

RUN cat > /usr/local/bin/glslc <<'EOF' && chmod +x /usr/local/bin/glslc
#!/usr/bin/env python3
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

args = sys.argv[1:]
output = None
input_file = None
forwarded = []
target_env = None

i = 0
while i < len(args):
    arg = args[i]
    if arg == "-o":
        output = args[i + 1]
        i += 2
        continue
    if arg in {"-mfmt=num", "-O"}:
        i += 1
        continue
    if arg.startswith("--target-env="):
        target_env = arg.split("=", 1)[1]
        i += 1
        continue
    if not arg.startswith("-") and input_file is None:
        input_file = arg
        i += 1
        continue
    forwarded.append(arg)
    i += 1

if input_file is None or output is None:
    raise SystemExit("glslc wrapper requires an input file and -o output")

with tempfile.TemporaryDirectory() as tmpdir:
    input_path = Path(input_file).resolve()
    temp_source = Path(tmpdir) / input_path.name
    spv_path = Path(tmpdir) / "shader.spv"
    source_text = input_path.read_text(encoding="utf-8")
    source_text = source_text.replace(
        "#extension GL_EXT_maximal_reconvergence : require\n",
        "",
    )
    source_text = source_text.replace(
        "#extension GL_EXT_shader_quad_control : require\n",
        "",
    )
    source_text = source_text.replace(", full_quads", "")
    source_text = source_text.replace("subgroupQuadAll(", "subgroupAll(")
    source_lines = source_text.splitlines(keepends=True)
    if source_lines and source_lines[0].lstrip().startswith("#version"):
        patched_source = (
            source_lines[0]
            + "#extension GL_GOOGLE_include_directive : require\n"
            + "".join(source_lines[1:])
        )
    else:
        patched_source = "#extension GL_GOOGLE_include_directive : require\n" + source_text
    temp_source.write_text(patched_source, encoding="utf-8")
    cmd = [
        "glslangValidator",
        "-V",
        str(temp_source),
        "-o",
        str(spv_path),
        "-I" + str(input_path.parent),
        "-I" + str(input_path.parent.parent),
        *forwarded,
    ]
    if target_env is not None:
        cmd.extend(["--target-env", target_env])
    subprocess.run(cmd, check=True)
    data = spv_path.read_bytes()
    if len(data) % 4 != 0:
        raise SystemExit("SPIR-V output length is not divisible by 4 bytes")
    words = struct.unpack("<" + "I" * (len(data) // 4), data)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(",\n".join(str(word) for word in words), encoding="utf-8")
EOF

COPY . /app

ENV ONNXRUNTIME_VERSION=1.25.0
RUN curl -L "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz" -o /tmp/onnxruntime.tgz \
    && mkdir -p /opt/onnxruntime \
    && tar -xzf /tmp/onnxruntime.tgz --strip-components=1 -C /opt/onnxruntime \
    && rm /tmp/onnxruntime.tgz \
    && ldconfig /opt/onnxruntime/lib || true

RUN mkdir -p /opt/lpips_models \
    && cd /app \
    && python -m core.rendering.export_lpips_onnx --net alex --width 320 --height 240 --output /opt/lpips_models/lpips_alex_320x240.onnx \
    && python -m core.rendering.export_lpips_onnx --net vgg --width 320 --height 240 --output /opt/lpips_models/lpips_vgg_320x240.onnx \
    && python -m core.rendering.export_lpips_onnx --net squeeze --width 320 --height 240 --output /opt/lpips_models/lpips_squeeze_320x240.onnx

RUN cmake -S /app/third_party/VkDiffGaussianRasterizer -B /opt/vkdiff_build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /opt/vkdiff_build --target ForwardRenderServer

ENV PYTHONUNBUFFERED=1
ENV SPLAT_PATH=/workspace/splat.ply
ENV RENDERER_BACKEND=vkdiff
ENV VKDIFF_SERVER_PATH=/opt/vkdiff_build/ForwardRenderServer
ENV VKDIFF_LPIPS_MODEL_DIR=/opt/lpips_models
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib:/opt/conda/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

EXPOSE 8000

CMD ["uvicorn", "core.rendering.api:app", "--host", "0.0.0.0", "--port", "8000"]
