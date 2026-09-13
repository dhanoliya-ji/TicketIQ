"""Pydantic models that define the HTTP contract.

Keeping these in one file means the whole API surface can be read in a single
screen, and FastAPI turns them into the OpenAPI documentation served at
``/docs`` automatically.
"""

from pydantic import BaseModel, Field

# The tiers a customer can be on. Anything else is rejected with a 422.
ALLOWED_TIERS = ["free", "pro", "enterprise"]


# ---------------------------------------------------------------------------
# POST /ticket
# ---------------------------------------------------------------------------
class TicketRequest(BaseModel):
    """A support ticket submitted by a customer."""

    subject: str = Field(min_length=1, max_length=300, description="The ticket subject line.")
    body: str = Field(min_length=1, max_length=5000, description="The free text ticket body.")
    customer_tier: str = Field(
        default="free",
        description="One of: free, pro, enterprise.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "subject": "Charged twice on order 4471",
                    "body": "My credit card was charged twice for the same monthly invoice.",
                    "customer_tier": "enterprise",
                }
            ]
        }
    }


class AspectSentimentModel(BaseModel):
    aspect: str
    score: float
    label: str
    mentions: int
    evidence: str


class KnowledgeSnippetModel(BaseModel):
    chunk_id: str
    source: str
    document_title: str
    heading: str
    text: str
    score: float


class ReasoningStepModel(BaseModel):
    step: int
    thought: str
    action: str
    action_input: dict
    observation: str


class ToolCallModel(BaseModel):
    tool: str
    arguments: dict
    output: dict
    summary: str


class PipelineConfigModel(BaseModel):
    name: str
    prompt_variant: str
    rag_top_k: int
    model: str


class TicketResponse(BaseModel):
    """Everything the pipeline produced for one ticket."""

    transaction_id: str
    category: str
    category_confidence: float
    aspect_sentiments: list[AspectSentimentModel]
    urgency_score: float
    urgency_bucket: str
    retrieved_knowledge: list[KnowledgeSnippetModel]
    action: str
    response_text: str
    reasoning_trace: list[ReasoningStepModel]
    tool_calls: list[ToolCallModel]
    pipeline_config: PipelineConfigModel
    config_selection_reason: str
    rl_state_key: str
    llm_backend: str
    latency_seconds: float
    stages_executed: list[str]
    stages_reused: list[str]


# ---------------------------------------------------------------------------
# POST /feedback
# ---------------------------------------------------------------------------
class FeedbackRequest(BaseModel):
    """Binary feedback on an answer, which drives the reinforcement learning loop."""

    transaction_id: str = Field(min_length=1)
    feedback_score: int = Field(ge=0, le=1, description="1 = helpful, 0 = unhelpful.")


class FeedbackResponse(BaseModel):
    transaction_id: str
    feedback_score: int
    latency_seconds: float
    reward: float
    state_key: str
    config_name: str
    updated_average_reward: float
    best_config_for_state: str


# ---------------------------------------------------------------------------
# GET /ticket/{transaction_id}/status
# ---------------------------------------------------------------------------
class StageStatusModel(BaseModel):
    stage: str
    status: str
    output: dict
    error: str
    duration_seconds: float


class TicketStatusResponse(BaseModel):
    transaction_id: str
    status: str
    request: dict
    completed_stages: int
    total_stages: int
    stages: list[StageStatusModel]
    not_started_stages: list[str]
    feedback_score: int | None
    reward: float | None


# ---------------------------------------------------------------------------
# Inspection endpoints
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    ready: bool
    llm_backend: str
    knowledge_chunks: int
    classifier_accuracy: float | None
    bandit_updates: int
