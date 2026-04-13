FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
      git wget vim build-essential \
      libgl1-mesa-glx libglib2.0-0 \
      && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt

# 1. requirements.txt 설치
RUN pip install --no-cache-dir -r requirements.txt

# 2. diffusers/xformers 버전 고정 (PyTorch 2.1.0 + CUDA 11.8 호환)
RUN pip install --no-cache-dir "diffusers==0.21.4" "transformers<4.36" "huggingface_hub<0.24"
RUN pip install --no-cache-dir xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu118

# 3. omegaconf (text2im.py), accelerate (모델 로딩 가속)
RUN pip install --no-cache-dir omegaconf accelerate

# 4. numpy<2를 마지막에 설치 (PyTorch 2.1.0은 numpy 1.x 필요, 다른 패키지가 2.x로 올릴 수 있으므로)
RUN pip install --no-cache-dir "numpy<2"

# 5. 검증
RUN python -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda)"
RUN python -c "import numpy; print('NumPy:', numpy.__version__)"

WORKDIR /workspace/XPaste
