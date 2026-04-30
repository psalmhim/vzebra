FROM python:3.12-slim

# System deps: SDL2 for pygame (headless), ffmpeg for video
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev \
    ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps in a separate layer so Docker cache avoids
# re-downloading PyTorch (~750MB) on every code push.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    grep -v '^torch' requirements.txt | pip install --no-cache-dir -r /dev/stdin

# Copy project code (invalidates cache only here — fast on subsequent builds)
COPY . .

ENV SDL_VIDEODRIVER=dummy \
    SDL_AUDIODRIVER=dummy \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

EXPOSE 5001
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:5001/api/status || exit 1

CMD ["python", "-m", "zebrav2.web.server"]
