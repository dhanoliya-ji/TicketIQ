"""The triage pipeline: the seven stages and the service that owns them.

The graph
---------
::

    level 0   classify_ticket        analyse_sentiment      (run in parallel)
                     \\                    /
    level 1              score_urgency
                              |
    level 2          select_configuration        <- the bandit chooses here
                              |
    level 3          retrieve_knowledge          <- uses the chosen top-K
                              |
    level 4               run_agent              <- uses the chosen prompt
                              |
    level 5           compose_response

``classify_ticket`` and ``analyse_sentiment`` both read only the raw ticket, so
they are independent and the engine runs them on two threads.  Everything
after that is a real dependency: urgency needs both of them, the bandit needs
urgency to know which state it is in, retrieval needs the top-K the bandit
picked, and the agent needs the retrieved knowledge.

Each stage returns a plain dictionary, which the engine persists.  That is why
the stages pass dictionaries around rather than objects.
"""

import time
import uuid

from app.agent.react_agent import TriageAgent
from app.llm.client import LlmClient
from app.llm.configs import ALL_CONFIGS, all_config_names, get_config
from app.ml.aspect_sentiment import AspectSentiment, AspectSentimentAnalyzer
from app.ml.classifier_service import TicketClassifierService
from app.ml.urgency import score_urgency, urgency_bucket
from app.rag.retriever import KnowledgeRetriever
from app.rl.bandit import EpsilonGreedyContextualBandit
from app.rl.state import build_state_key, compute_reward
from app.settings import SETTINGS
from app.workflow.dag import Stage, StageContext, WorkflowEngine
from app.workflow.state_store import STATUS_COMPLETED, STATUS_FAILED, WorkflowStateStore

# Stage names, used as dictionary keys, database keys and API field names.
STAGE_CLASSIFY = "classify_ticket"
STAGE_SENTIMENT = "analyse_sentiment"
STAGE_URGENCY = "score_urgency"
STAGE_SELECT_CONFIG = "select_configuration"
STAGE_RETRIEVE = "retrieve_knowledge"
STAGE_AGENT = "run_agent"
STAGE_COMPOSE = "compose_response"


class TriageService:
    """Owns every component and exposes the three operations the API needs."""

    def __init__(self) -> None:
        SETTINGS.ensure_var_dir()

        # --- components ---
        self.classifier = TicketClassifierService(
            dataset_path=SETTINGS.tickets_file,
            test_fraction=SETTINGS.test_split,
            seed=SETTINGS.random_seed,
        )
        self.sentiment_analyzer = AspectSentimentAnalyzer()
        self.retriever = KnowledgeRetriever(SETTINGS.knowledge_base_dir)
        self.llm_client = LlmClient()
        self.agent = TriageAgent(self.llm_client)
        self.bandit = EpsilonGreedyContextualBandit(
            actions=all_config_names(),
            epsilon=SETTINGS.bandit_epsilon,
            seed=SETTINGS.random_seed,
        )
        self.store = WorkflowStateStore(SETTINGS.workflow_db_file)

        # --- the workflow graph ---
        self.engine = WorkflowEngine(self._build_stages(), self.store)

        self.is_ready = False

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------
    def startup(self) -> None:
        """Train the classifier, index the knowledge base, reload the bandit."""
        self.classifier.train()
        self.retriever.load()
        # Learning from previous runs is restored if a state file exists.
        self.bandit.load(SETTINGS.bandit_state_file)
        self.is_ready = True

    # ------------------------------------------------------------------
    # Stage definitions
    # ------------------------------------------------------------------
    def _build_stages(self) -> list[Stage]:
        return [
            Stage(
                name=STAGE_CLASSIFY,
                depends_on=[],
                run=self._stage_classify,
                description="Naive Bayes category prediction from the ticket text.",
            ),
            Stage(
                name=STAGE_SENTIMENT,
                depends_on=[],
                run=self._stage_sentiment,
                description="Per-aspect sentiment scoring with VADER.",
            ),
            Stage(
                name=STAGE_URGENCY,
                depends_on=[STAGE_CLASSIFY, STAGE_SENTIMENT],
                run=self._stage_urgency,
                description="Weighted urgency score from category, sentiment and tier.",
            ),
            Stage(
                name=STAGE_SELECT_CONFIG,
                depends_on=[STAGE_CLASSIFY, STAGE_URGENCY],
                run=self._stage_select_configuration,
                description="Contextual bandit picks the prompt variant and RAG top-K.",
            ),
            Stage(
                name=STAGE_RETRIEVE,
                depends_on=[STAGE_CLASSIFY, STAGE_SELECT_CONFIG],
                run=self._stage_retrieve,
                description="Cosine similarity search over the knowledge base.",
            ),
            Stage(
                name=STAGE_AGENT,
                depends_on=[STAGE_CLASSIFY, STAGE_URGENCY, STAGE_SELECT_CONFIG, STAGE_RETRIEVE],
                run=self._stage_agent,
                description="ReAct loop: tools, decision and the drafted reply.",
            ),
            Stage(
                name=STAGE_COMPOSE,
                depends_on=[STAGE_AGENT],
                run=self._stage_compose,
                description="Final customer facing response text.",
            ),
        ]

    def _stage_classify(self, context: StageContext) -> dict:
        subject = context.inputs["subject"]
        body = context.inputs["body"]
        return self.classifier.classify(subject, body).as_dict()

    def _stage_sentiment(self, context: StageContext) -> dict:
        subject = context.inputs["subject"]
        body = context.inputs["body"]
        aspects = self.sentiment_analyzer.analyse(subject, body)

        aspect_dicts = []
        for aspect in aspects:
            aspect_dicts.append(aspect.as_dict())
        return {"aspects": aspect_dicts}

    def _stage_urgency(self, context: StageContext) -> dict:
        category = context.output_of(STAGE_CLASSIFY)["category"]
        aspect_dicts = context.output_of(STAGE_SENTIMENT)["aspects"]
        customer_tier = context.inputs["customer_tier"]

        # Rebuild the objects the scorer expects from the persisted dictionaries.
        aspects = []
        for aspect_dict in aspect_dicts:
            aspects.append(
                AspectSentiment(
                    aspect=aspect_dict["aspect"],
                    score=float(aspect_dict["score"]),
                    mentions=int(aspect_dict["mentions"]),
                    evidence=aspect_dict["evidence"],
                )
            )

        score = score_urgency(category, aspects, customer_tier)
        return {"urgency_score": round(score, 4), "urgency_bucket": urgency_bucket(score)}

    def _stage_select_configuration(self, context: StageContext) -> dict:
        category = context.output_of(STAGE_CLASSIFY)["category"]
        bucket = context.output_of(STAGE_URGENCY)["urgency_bucket"]
        customer_tier = context.inputs["customer_tier"]

        state_key = build_state_key(category, bucket, customer_tier)
        action_name, selection_reason = self.bandit.select_action(state_key)
        config = get_config(action_name)

        return {
            "state_key": state_key,
            "selection_reason": selection_reason,
            "config": config.as_dict(),
        }

    def _stage_retrieve(self, context: StageContext) -> dict:
        category = context.output_of(STAGE_CLASSIFY)["category"]
        config = context.output_of(STAGE_SELECT_CONFIG)["config"]
        top_k = int(config["rag_top_k"])

        hits = self.retriever.retrieve(
            subject=context.inputs["subject"],
            body=context.inputs["body"],
            category=category,
            top_k=top_k,
        )

        snippets = []
        for hit in hits:
            snippets.append(hit.as_dict())
        return {"top_k": top_k, "snippets": snippets}

    def _stage_agent(self, context: StageContext) -> dict:
        config_dict = context.output_of(STAGE_SELECT_CONFIG)["config"]
        config = get_config(str(config_dict["name"]))

        agent_result = self.agent.run(
            subject=context.inputs["subject"],
            body=context.inputs["body"],
            customer_tier=context.inputs["customer_tier"],
            category=context.output_of(STAGE_CLASSIFY)["category"],
            urgency_bucket=context.output_of(STAGE_URGENCY)["urgency_bucket"],
            snippets=context.output_of(STAGE_RETRIEVE)["snippets"],
            config=config,
        )

        output = agent_result.as_dict()
        output["response_text"] = agent_result.response_text
        return output

    def _stage_compose(self, context: StageContext) -> dict:
        agent_output = context.output_of(STAGE_AGENT)
        return {
            "response_text": agent_output["response_text"],
            "action": agent_output["decision"],
        }

    # ------------------------------------------------------------------
    # Operation 1: handle a ticket
    # ------------------------------------------------------------------
    def handle_ticket(self, subject: str, body: str, customer_tier: str) -> dict:
        """Run the full pipeline for one ticket and return the API payload."""
        transaction_id = "tx-" + uuid.uuid4().hex[:12]
        request = {"subject": subject, "body": body, "customer_tier": customer_tier}

        self.store.create_ticket(transaction_id, request)

        # The latency measured here is what the reward function penalises, so
        # it must cover the whole pipeline, not just the model call.
        started_at = time.perf_counter()
        run_result = self.engine.run(transaction_id, request)
        latency_seconds = time.perf_counter() - started_at

        if not run_result.succeeded:
            self.store.finish_ticket(
                transaction_id=transaction_id,
                status=STATUS_FAILED,
                result={"failed_stage": run_result.failed_stage, "error": run_result.error},
                config_name="",
                state_key="",
                latency_seconds=latency_seconds,
            )
            raise PipelineError(transaction_id, run_result.failed_stage, run_result.error)

        outputs = run_result.outputs
        classification = outputs[STAGE_CLASSIFY]
        sentiment = outputs[STAGE_SENTIMENT]
        urgency = outputs[STAGE_URGENCY]
        selection = outputs[STAGE_SELECT_CONFIG]
        retrieval = outputs[STAGE_RETRIEVE]
        agent_output = outputs[STAGE_AGENT]
        composed = outputs[STAGE_COMPOSE]

        response = {
            "transaction_id": transaction_id,
            "category": classification["category"],
            "category_confidence": classification["confidence"],
            "aspect_sentiments": sentiment["aspects"],
            "urgency_score": urgency["urgency_score"],
            "urgency_bucket": urgency["urgency_bucket"],
            "retrieved_knowledge": retrieval["snippets"],
            "action": composed["action"],
            "response_text": composed["response_text"],
            "reasoning_trace": agent_output["reasoning_trace"],
            "tool_calls": agent_output["tool_calls"],
            "pipeline_config": selection["config"],
            "config_selection_reason": selection["selection_reason"],
            "rl_state_key": selection["state_key"],
            "llm_backend": agent_output["llm_backend"],
            "latency_seconds": round(latency_seconds, 4),
            "stages_executed": run_result.executed_stages,
            "stages_reused": run_result.reused_stages,
        }

        self.store.finish_ticket(
            transaction_id=transaction_id,
            status=STATUS_COMPLETED,
            result=response,
            config_name=str(selection["config"]["name"]),
            state_key=str(selection["state_key"]),
            latency_seconds=latency_seconds,
        )
        return response

    # ------------------------------------------------------------------
    # Operation 2: learn from feedback
    # ------------------------------------------------------------------
    def handle_feedback(self, transaction_id: str, feedback_score: int) -> dict:
        """Turn one piece of feedback into a bandit update.

        The reward needs the latency and the action that produced the answer,
        both of which were stored when the ticket finished - which is why
        feedback can safely arrive minutes later or after a restart.
        """
        ticket = self.store.get_ticket(transaction_id)
        if ticket is None:
            raise UnknownTransactionError(transaction_id)

        if ticket["status"] != STATUS_COMPLETED or not ticket["config_name"]:
            raise FeedbackNotAcceptedError(
                transaction_id, "the pipeline for this ticket did not complete successfully"
            )

        # Feedback is applied once. A second call would teach the bandit the
        # same lesson twice and quietly bias the averages.
        if ticket["feedback_score"] is not None:
            raise FeedbackNotAcceptedError(
                transaction_id, "feedback has already been recorded for this transaction"
            )

        latency_seconds = float(ticket["latency_seconds"] or 0.0)
        reward = compute_reward(feedback_score, latency_seconds)

        state_key = str(ticket["state_key"])
        config_name = str(ticket["config_name"])

        self.bandit.update(state_key, config_name, reward)
        self.bandit.save(SETTINGS.bandit_state_file)
        self.store.record_feedback(transaction_id, feedback_score, reward)

        return {
            "transaction_id": transaction_id,
            "feedback_score": feedback_score,
            "latency_seconds": round(latency_seconds, 4),
            "reward": round(reward, 4),
            "state_key": state_key,
            "config_name": config_name,
            "updated_average_reward": round(
                self.bandit.statistics[state_key][config_name].average_reward, 4
            ),
            "best_config_for_state": self.bandit.best_action(state_key),
        }

    # ------------------------------------------------------------------
    # Operation 3: inspect the pipeline
    # ------------------------------------------------------------------
    def get_status(self, transaction_id: str) -> dict:
        """Report the real per-stage state held by the workflow engine."""
        ticket = self.store.get_ticket(transaction_id)
        if ticket is None:
            raise UnknownTransactionError(transaction_id)

        records = self.store.get_stages(transaction_id)

        stage_dicts = []
        completed_count = 0
        for record in records:
            stage_dicts.append(record.as_dict())
            if record.status == STATUS_COMPLETED:
                completed_count = completed_count + 1

        # Stages that have no row yet have simply not been reached.
        all_stage_names = list(self.engine.stages_by_name.keys())
        seen_names = [record.stage_name for record in records]
        pending_names = []
        for name in all_stage_names:
            if name not in seen_names:
                pending_names.append(name)

        return {
            "transaction_id": transaction_id,
            "status": ticket["status"],
            "request": ticket["request"],
            "completed_stages": completed_count,
            "total_stages": len(all_stage_names),
            "stages": stage_dicts,
            "not_started_stages": pending_names,
            "feedback_score": ticket["feedback_score"],
            "reward": ticket["reward"],
        }

    # ------------------------------------------------------------------
    # Extra inspection endpoints
    # ------------------------------------------------------------------
    def rl_statistics(self) -> dict:
        snapshot = self.bandit.snapshot()
        snapshot["total_updates"] = self.bandit.total_pulls()
        snapshot["configurations"] = [config.as_dict() for config in ALL_CONFIGS]
        return snapshot

    def workflow_description(self) -> list[dict[str, object]]:
        return self.engine.describe()

    def health(self) -> dict:
        return {
            "ready": self.is_ready,
            "llm_backend": self.llm_client.active_backend,
            "knowledge_chunks": self.retriever.chunk_count() if self.retriever.is_loaded else 0,
            "classifier_accuracy": self.classifier.evaluation.get("accuracy"),
            "bandit_updates": self.bandit.total_pulls(),
        }


# ---------------------------------------------------------------------------
# Errors the API turns into HTTP responses
# ---------------------------------------------------------------------------
class PipelineError(Exception):
    """A stage failed, so the ticket has no answer."""

    def __init__(self, transaction_id: str, failed_stage: str, error: str) -> None:
        super().__init__("stage '" + failed_stage + "' failed: " + error)
        self.transaction_id = transaction_id
        self.failed_stage = failed_stage
        self.error = error


class UnknownTransactionError(Exception):
    """No ticket exists with that transaction id."""

    def __init__(self, transaction_id: str) -> None:
        super().__init__("unknown transaction id: " + transaction_id)
        self.transaction_id = transaction_id


class FeedbackNotAcceptedError(Exception):
    """The transaction exists but feedback cannot be applied to it."""

    def __init__(self, transaction_id: str, reason: str) -> None:
        super().__init__(reason)
        self.transaction_id = transaction_id
        self.reason = reason
