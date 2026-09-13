# syntax=docker/dockerfile:1.6

# ---------------------------------------------------------------
# Stage 1: build
# ---------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for NumPy/SciPy/Numba
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging metadata first (better layer caching)
COPY pyproject.toml README.md ./
COPY eer/ ./eer/
COPY requirements.txt ./

# Install into a virtualenv we can copy to the runtime stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip wheel setuptools \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

# ---------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Epistemic Ecology Runtime"
LABEL org.opencontainers.image.description="Variational belief relaxation on epistemic graphs"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.source="https://github.com/myosettnaing1992-alt/Epistemic-Ecology-Runtime"

WORKDIR /app

# Copy the pre-built venv
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy source + tests + benchmarks
COPY eer/ ./eer/
COPY tests/ ./tests/
COPY benchmarks/ ./benchmarks/
COPY pyproject.toml README.md ./

# Default command: run the test suite
CMD ["pytest", "-v", "tests/"]
