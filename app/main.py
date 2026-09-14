"""The FastAPI application: three required endpoints plus inspection and the UI.

Run it locally with::

    uvicorn app.main:app --reload

Then:

* http://localhost:8000/      - the live console (a demo UI for the whole pipeline)
* http://localhost:8000/docs  - interactive OpenAPI documentation

The console is a plain HTML/CSS/JavaScript page served by this same
application, so it shares an origin with the API and needs no CORS setup, no
Node tooling and no extra dependency.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import (
    ALLOWED_TIERS,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    TicketRequest,
    TicketResponse,
    TicketStatusResponse,
)
from app.workflow.pipeline import (
    FeedbackNotAcceptedError,
    PipelineError,
    RetryNotNeededError,
    TriageService,
    UnknownTransactionError,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("ticketiq")

# The folder holding the live console (index.html, styles.css, app.js).
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Train the classifier and index the knowledge base before serving traffic.

    Startup work belongs here rather than at import time so that importing the
    module (in tests, for example) stays cheap and side-effect free.
    """
    service.startup()
    logger.info(
        "TicketIQ ready: llm_backend=%s knowledge_chunks=%s classifier_accuracy=%s",
        service.llm_client.active_backend,
        service.retriever.chunk_count(),
        service.classifier.evaluation.get("accuracy"),
    )
    yield


app = FastAPI(
    lifespan=lifespan,
    title="TicketIQ",
    version="1.0.0",
    description=(
        "Self-optimizing support ticket triage agent: classical ML classification, "
        "aspect sentiment, RAG, a ReAct agent and an online contextual bandit, "
        "orchestrated by a resumable workflow engine."
    ),
)

# Serve the console's assets. This is mounted under /static rather than at the
# root so it can never shadow an API route.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# One service instance for the whole process. It holds the trained classifier,
# the knowledge index and the bandit statistics, all of which are expensive to
# build and safe to share between requests.
service = TriageService()


# ---------------------------------------------------------------------------
# The live console
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def get_console() -> FileResponse:
    """Triage: submit a ticket and read the decision."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/performance", include_in_schema=False)
def get_performance() -> FileResponse:
    """Performance: classification quality, routing, and the pipeline.

    A separate page rather than another panel on the ticket view, because it
    answers a different question for a different reader - "is this system any
    good?" rather than "what should happen to this ticket?" - and it is the
    only place with more data than one ticket's slice.
    """
    return FileResponse(STATIC_DIR / "performance.html")


# ---------------------------------------------------------------------------
# Required endpoints
# ---------------------------------------------------------------------------
@app.post("/ticket", response_model=TicketResponse)
def post_ticket(request: TicketRequest) -> dict:
    """Run the full triage pipeline for one ticket."""
    if request.customer_tier not in ALLOWED_TIERS:
        raise HTTPException(
            status_code=422,
            detail="customer_tier must be one of: " + ", ".join(ALLOWED_TIERS),
        )

    try:
        result = service.handle_ticket(
            subject=request.subject,
            body=request.body,
            customer_tier=request.customer_tier,
        )
    except PipelineError as error:
        # The transaction id is still returned so the caller can inspect which
        # stage failed through the status endpoint.
        logger.error("pipeline failed at stage %s: %s", error.failed_stage, error.error)
        raise HTTPException(
            status_code=500,
            detail={
                "message": "the triage pipeline failed",
                "transaction_id": error.transaction_id,
                "failed_stage": error.failed_stage,
                "error": error.error,
            },
        ) from error

    logger.info(
        "ticket %s category=%s urgency=%s config=%s action=%s latency=%.3fs",
        result["transaction_id"],
        result["category"],
        result["urgency_bucket"],
        result["pipeline_config"]["name"],
        result["action"],
        result["latency_seconds"],
    )
    return result


@app.post("/feedback", response_model=FeedbackResponse)
def post_feedback(request: FeedbackRequest) -> dict:
    """Record binary feedback and update the bandit with the resulting reward."""
    try:
        result = service.handle_feedback(request.transaction_id, request.feedback_score)
    except UnknownTransactionError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except FeedbackNotAcceptedError as error:
        raise HTTPException(status_code=409, detail=error.reason) from error

    logger.info(
        "feedback %s score=%s reward=%.3f state=%s config=%s",
        request.transaction_id,
        request.feedback_score,
        result["reward"],
        result["state_key"],
        result["config_name"],
    )
    return result


@app.post("/ticket/{transaction_id}/retry", response_model=TicketResponse)
def post_ticket_retry(transaction_id: str) -> dict:
    """Re-run a failed pipeline, reusing every stage that already succeeded.

    This is the resumable half of requirement 6 made usable: the response still
    carries ``stages_reused`` and ``stages_executed``, so it is visible that the
    upstream work was not repeated.
    """
    try:
        result = service.retry_ticket(transaction_id)
    except UnknownTransactionError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RetryNotNeededError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except PipelineError as error:
        logger.error("retry failed again at stage %s: %s", error.failed_stage, error.error)
        raise HTTPException(
            status_code=500,
            detail={
                "message": "the triage pipeline failed again",
                "transaction_id": error.transaction_id,
                "failed_stage": error.failed_stage,
                "error": error.error,
            },
        ) from error

    logger.info(
        "retry %s reused=%s executed=%s",
        transaction_id,
        result["stages_reused"],
        result["stages_executed"],
    )
    return result


@app.get("/ticket/{transaction_id}/status", response_model=TicketStatusResponse)
def get_ticket_status(transaction_id: str) -> dict:
    """Report which pipeline stages have completed, straight from the state store."""
    try:
        return service.get_status(transaction_id)
    except UnknownTransactionError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


# ---------------------------------------------------------------------------
# Inspection endpoints (not required, but they make the system reviewable)
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def get_health() -> dict:
    """Is the service ready, and which language model backend is live?"""
    return service.health()


@app.get("/rl/stats")
def get_rl_stats() -> dict:
    """Everything the bandit has learned so far, per state and per action."""
    return service.rl_statistics()


@app.get("/workflow/graph")
def get_workflow_graph() -> dict:
    """The pipeline DAG, level by level, showing what runs in parallel."""
    return {"levels": service.workflow_description()}


@app.get("/ml/report")
def get_ml_report() -> dict:
    """The held-out evaluation of the ticket classifier."""
    return {
        "evaluation": service.classifier.evaluation,
        "confusion_matrix": service.classifier.confusion,
    }
