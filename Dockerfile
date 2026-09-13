# ---------------------------------------------------------------------------
# TicketIQ service image.
#
# Build:  docker build -t ticketiq .
# Run  :  docker run --rm -p 8000:8000 ticketiq
#
# The image does not contain a language model. By default the service runs
# with TICKETIQ_LLM_BACKEND=auto, which probes Ollama and falls back to the
# offline template writer, so the container is useful on its own. To use a
# real model, point it at an Ollama server on the host:
#
#   docker run --rm -p 8000:8000 \
#     -e TICKETIQ_OLLAMA_URL=http://host.docker.internal:11434 ticketiq
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# Python behaves better in containers with these two settings: no .pyc files
# on disk, and log output that is not buffered away when the container stops.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /service

# Dependencies are installed first and in their own layer, so that editing
# application code does not invalidate the cached pip install.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Application code, the knowledge base and the labelled dataset.
COPY app ./app
COPY data ./data

# Runtime state (SQLite workflow state, bandit statistics) lives here. It is
# a separate directory so it can be mounted as a volume to survive restarts.
RUN mkdir -p /service/var

# Run as a non-root user: the service never needs to write outside /service.
RUN useradd --create-home --uid 1000 ticketiq \
    && chown -R ticketiq:ticketiq /service
USER ticketiq

EXPOSE 8000

# A failing health check means the classifier or knowledge base did not load.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
