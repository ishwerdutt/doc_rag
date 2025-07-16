# syntax=docker/dockerfile:1

FROM python:3.13-slim AS base


WORKDIR /app


# Builder stage: install dependencies in a venv
FROM base AS builder

ARG http_proxy
ARG https_proxy
ENV http_proxy=$http_proxy
ENV https_proxy=$https_proxy

# Install system dependencies for common Python packages (e.g., for numpy, torch, faiss, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        git && \
    rm -rf /var/lib/apt/lists/*


# Copy only requirements.txt first for better cache usage
COPY --link requirements.txt ./

# Create venv and install dependencies using pip cache
RUN python -m venv .venv && \
    .venv/bin/pip install --upgrade pip && \
    .venv/bin/pip install -r requirements.txt

# Copy the rest of the application code (excluding .git, .env, etc.)
COPY --link . .

# Final stage: minimal image with app code and venv
FROM base AS final

# Create a non-root user
RUN useradd -m appuser

WORKDIR /app

# Copy venv and app code from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

# Set environment so venv is used by default
ENV PATH="/app/.venv/bin:$PATH"

# Create data directories and set permissions
RUN mkdir -p /app/data/pdfs /app/data/faiss_index && chmod -R 777 /app/data

# Expose the port Flask/Gunicorn will listen on
EXPOSE 5000

# Use non-root user
USER appuser

# Default command (can be overridden)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
