FROM python:3.11-slim

LABEL maintainer="Myo Sett Naing <you@example.com>"
LABEL description="Epistemic Ecology Runtime (EER) reproducible environment"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/
  
RUN pip install --upgrade pip setuptools wheel

COPY pyproject.toml README.md LICENSE CITATION.cff ./
COPY eer/ ./eer/
COPY tests/ ./tests/
COPY benchmarks/ ./benchmarks/

RUN pip install -e ".[dev,bench]"

RUN python -c "import eer; print('EER', eer.__version__)"

CMD ["pytest", "-v", "tests/"]
