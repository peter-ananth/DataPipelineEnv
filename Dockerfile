FROM python:3.11-slim

# Build-time labels
LABEL org.opencontainers.image.title="DataPipelineEnv"
LABEL org.opencontainers.image.description="OpenEnv for debugging data pipelines — CSV cleaning + SQL fixing"
LABEL org.opencontainers.image.version="1.0.0"

WORKDIR /app

# Install system dependencies (curl for health check)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY inference.py .
COPY openenv.yaml .

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=7860

EXPOSE 7860

# Health check — required for HF Spaces and Docker orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
