FROM python:3.12-slim

# Set build argument for CUDA support (optional)
ARG ENABLE_CUDA=false

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Install sentence-transformers with CUDA support if enabled
RUN if [ "$ENABLE_CUDA" = "true" ]; then \
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 && \
        pip install -r requirements.txt; \
    else \
        pip install -r requirements.txt; \
    fi

# Copy application code
COPY . .

# Create reports directory
RUN mkdir -p /app/reports

# Expose port
EXPOSE 8000

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
