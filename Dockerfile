# Quant Assistant -- container image for the FastAPI web app.
#
#   docker build -t quant-assistant .
#   docker run --rm -p 8000:8000 -e ANTHROPIC_API_KEY=sk-... quant-assistant
#   # open http://127.0.0.1:8000
#
# The image bakes a retrieval index from the bundled synthetic sample data using
# the dependency-free hashing backend, so the container serves the app end to end
# without any build-time secret. A Claude API key is only needed at run time for
# the /api/ask endpoint (pass it with -e ANTHROPIC_API_KEY=...).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    EMBEDDINGS_BACKEND=hashing \
    QA_INDEX=/app/index.pkl

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the retrieval index from the bundled sample data (no API key required).
RUN python -m quant_assistant.cli ingest \
      --notes data/sample/research_notes.md \
      --data data/sample/prices.csv

EXPOSE 8000

CMD ["uvicorn", "web.server:app", "--host", "0.0.0.0", "--port", "8000"]
