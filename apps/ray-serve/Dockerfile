# Start with Ray's GPU base image
FROM rayproject/ray:2.31.0-py310-gpu

RUN pip install ray[serve]
# Install the PyTorch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install transformers python-multipart kornia uvicorn fastapi timm
