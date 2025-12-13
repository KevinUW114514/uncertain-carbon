# ------------------------------------------------------
# Stage 1: Base image
# ------------------------------------------------------
FROM python:3.11-slim

# Set environment vars for Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    THREADS=2

# Set working directory
WORKDIR /app

COPY requirements.txt .
COPY requirements-cpu.txt .

# Install OS dependencies (Pillow needs image libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements-cpu.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY ml_object_detect.py app.py
COPY caching_models.py .

RUN python caching_models.py

# Expose Flask port
EXPOSE 8124

# Default command to run the Flask app
CMD ["bash", "-lc", "exec gunicorn --threads ${THREADS:-2} -b 0.0.0.0:8124 app:app"]
