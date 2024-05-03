FROM nvidia/cuda:12.3.2-base-ubuntu22.04

# Set the working directory
WORKDIR /app

# Copy your code to the container
COPY . /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch
RUN pip3 install torch torchvision
RUN pip3 install -r requirements.txt