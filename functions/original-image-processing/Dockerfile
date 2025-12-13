# ------------------------------------------------------
# Stage 1: Base image
# ------------------------------------------------------
FROM python:3.11-slim

# Set environment vars for Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

COPY requirements.txt .

# Install OS dependencies (Pillow needs image libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the application code
COPY ml_image_processing.py ./app.py

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask port
EXPOSE 8123

# Default command to run the Flask app
CMD ["python", "app.py"]
