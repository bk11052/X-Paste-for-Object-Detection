FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
      git wget vim build-essential \
      libgl1-mesa-glx libglib2.0-0 \
      && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt

# 1. numpy를 1.x로 고정 (PyTorch 2.1.0이 numpy 1.x로 빌드됨)
RUN pip install --no-cache-dir "numpy<2"

# 2. requirements.txt 설치 (기존 파이토치 2.1.0 유지)
RUN pip install --no-cache-dir -r requirements.txt

# 3. diffusers/xformers 버전 고정 (PyTorch 2.1.0 호환)
#    - diffusers 0.25.x: torch.xpu 미요구, PyTorch 2.1 호환 마지막 안정 버전대
#    - xformers 0.0.22.post7: PyTorch 2.1.0 전용
RUN pip install --no-cache-dir "diffusers==0.25.1" "transformers<4.40" xformers==0.0.22.post7

# 4. omegaconf 명시 설치 (text2im.py에서 사용)
RUN pip install --no-cache-dir omegaconf

# 4. 파이토치 버전 확인
RUN python -c "import torch; print('*** Torch Version:', torch.__version__); print('*** CUDA Version:', torch.version.cuda)"

WORKDIR /workspace/XPaste
