FROM python:3.11-slim

# Install system dependencies
# libgomp1 is needed for PaddlePaddle
# libgl1-mesa-glx/libglib2.0-0 are needed for OpenCV
# ffmpeg is needed for video processing
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Preload PaddleOCR models (Manual download to avoid build crash)
COPY manual_download.py .
RUN python manual_download.py && rm manual_download.py

# Create output directory
RUN mkdir -p output

# Copy source code
COPY . .

# Expose port
EXPOSE 8000

# Fix for "primitive descriptor" error on some CPUs (Disables MKLDNN optimizations)
ENV FLAGS_use_mkldnn=0
ENV FLAGS_enable_mkldnn=0

# Start command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
