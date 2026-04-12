FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
      git wget vim build-essential \
      libgl1-mesa-glx libglib2.0-0 \
      && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt

# 1. requirements.txt 설치 (기존 파이토치 2.1.0 유지)
RUN pip install --no-cache-dir -r requirements.txt

# 2. xformers는 파이토치 2.1.0과 호환되는 버전으로 고정
RUN pip install --no-cache-dir diffusers transformers xformers==0.0.22.post7

# 3. omegaconf 명시 설치 (text2im.py에서 사용)
RUN pip install --no-cache-dir omegaconf

# 4. 파이토치 버전 확인
RUN python -c "import torch; print('*** Torch Version:', torch.__version__); print('*** CUDA Version:', torch.version.cuda)"

WORKDIR /workspace/XPaste
