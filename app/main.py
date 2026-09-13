"""The FastAPI application: three required endpoints plus three for inspection.

Run it locally with::

    uvicorn app.main:app --reload

Interactive documentation is then served at http://localhost:8000/docs
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

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
    TriageService,
    UnknownTransactionError,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("ticketiq")


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

# One service instance for the whole process. It holds the trained classifier,
# the knowledge index and the bandit statistics, all of which are expensive to
# build and safe to share between requests.
service = TriageService()


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
