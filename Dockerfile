# Use Python 3.11 base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set HuggingFace cache directory
ENV HF_HOME=/app/.cache/huggingface

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy local data files (questions and characters)
COPY guess_game_models_enhanced_v2 ./guess_game_models_enhanced_v2

# Copy download script and pre-download models from HuggingFace
COPY download_models.py .
RUN python download_models.py

# Copy the rest of the application
COPY . .

# Expose port (Render will set $PORT environment variable)
EXPOSE $PORT

# Health check (optional but recommended)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:'+os.environ.get('PORT', '10000')+'/health')"

# Start the application with Gunicorn
# - Increased timeout for model loading
# - Single worker to reduce memory usage
# - Access log to stdout for debugging
CMD gunicorn --bind 0.0.0.0:${PORT} \
    --workers 1 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    app:app