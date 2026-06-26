FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
      git wget vim build-essential \
      libgl1-mesa-glx libglib2.0-0 \
      && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt

# 1. Base requirements (ultralytics, opencv, scikit-image, diffusers, transformers, ...)
RUN pip install --no-cache-dir -r requirements.txt

# 2. Pin diffusers/transformers/xformers for PyTorch 2.1.0 + CUDA 11.8.
#    transformers 4.40.2: SegFormer support, sweet spot for torch 2.1.
#    tokenizers 0.19.1: matching dependency for transformers 4.40.
RUN pip install --no-cache-dir \
        "diffusers==0.21.4" \
        "transformers==4.40.2" \
        "tokenizers==0.19.1" \
        "huggingface_hub<0.24" \
        matplotlib
RUN pip install --no-cache-dir xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu118

# 3. omegaconf (text2im.py), accelerate (faster model loading)
RUN pip install --no-cache-dir omegaconf accelerate

# 4. Reinstall numpy<2 last (PyTorch 2.1.0 needs numpy 1.x; other packages may bump it to 2.x)
RUN pip install --no-cache-dir "numpy<2"

# 5. Sanity check
RUN python -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda)"
RUN python -c "import numpy; print('NumPy:', numpy.__version__)"

WORKDIR /workspace/XPaste
