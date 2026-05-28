FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# spaCy model is not on PyPI
RUN python -m spacy download en_core_web_sm

COPY src/ src/
COPY dashboard/ dashboard/
COPY monitoring/ monitoring/

RUN mkdir -p data models reports

RUN useradd -m -u 1000 intellidesk
USER intellidesk

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]