# Official PyTorch 2.7.1 image with cuda 11.8
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

# Metadata
LABEL maintainer="francescosechi2505@gmail.com"
LABEL description="MS-LENSES: Multiple Sclerosis Lesion Ensemble Segmentation System"

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Copy mslenses directory
COPY mslenses/ /app

# Download python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set entrypoint
ENTRYPOINT ["python", "Ensemble.py"]
CMD ["--help"]