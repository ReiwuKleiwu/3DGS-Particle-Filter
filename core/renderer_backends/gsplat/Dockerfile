FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /app

ARG TORCH_CUDA_ARCH_LIST=8.6
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libglm-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml
COPY core /app/core

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install ninja \
    && python -m pip install .[rendering] \
    && python -m pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git

COPY . /app

ENV PYTHONUNBUFFERED=1
ENV SPLAT_PATH=/workspace/splat.ply
ENV RENDERER_BACKEND=gsplat

EXPOSE 8000

CMD ["uvicorn", "core.rendering.api:app", "--host", "0.0.0.0", "--port", "8000"]
