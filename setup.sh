#!/bin/bash
# Install system dependencies
apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libjpeg-dev \
    zlib1g-dev

# Install Python dependencies
pip install -r requirements.txt
