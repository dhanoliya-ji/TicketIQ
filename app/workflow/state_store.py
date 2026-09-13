"""Durable per-ticket, per-stage state, stored in SQLite.

Two tables:

``tickets``     one row per transaction: the request, the final result, the
                configuration the bandit chose, the measured latency and the
                feedback once it arrives.
``stage_runs``  one row per (transaction, stage): its status, its output and
                how long it took.

Why SQLite and not an in-memory dictionary: ``GET /ticket/{id}/status`` has to
show real state while a pipeline is still running, a failed stage has to be
re-runnable without repeating the stages that already succeeded, and feedback
can arrive minutes after the answer - possibly after a restart.  A file that
survives the process is the simplest thing that satisfies all three, and
SQLite is in the standard library.

Every method opens its own short-lived connection and takes a lock, because
the workflow engine runs stages on several threads at once.
"""

import json
import sqlite3
import threading
import time
from pathlib import Path

# The lifecycle of a single stage.
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"  # an upstream stage failed, so this never started


class StageRecord:
    """One stored stage row."""

    def __init__(
        self,
        stage_name: str,
        status: str,
        output: dict,
        error: str,
        started_at: float,
        finished_at: float,
    ) -> None:
        self.stage_name = stage_name
        self.status = status
        self.output = output
        self.error = error
        self.started_at = started_at
        self.finished_at = finished_at

    @property
    def duration_seconds(self) -> float:
        if self.started_at <= 0.0 or self.finished_at <= 0.0:
            return 0.0
        return self.finished_at - self.started_at

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage_name,
            "status": self.status,
            "output": self.output,
            "error": self.error,
            "duration_seconds": round(self.duration_seconds, 4),
        }


class WorkflowStateStore:
    """All reads and writes of workflow state go through this class."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.lock = threading.Lock()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        """Open a connection and guarantee the schema exists on it.

        The ``CREATE TABLE IF NOT EXISTS`` statements run on *every* connection
        rather than once at start-up, which makes the store self-healing.

        This is not theoretical: if the database file is deleted, moved or
        restored while the service is running - a cleaned ``var/`` folder, a
        container volume remounted, an operator tidying up - SQLite silently
        creates a fresh empty file on the next connection, and a store that had
        only created its tables once would then raise "no such table" on every
        request until the process was restarted. Two IF NOT EXISTS statements
        against an existing schema cost microseconds, which is a price worth
        paying to never serve that error.
        """
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.database_path))
        connection.row_factory = sqlite3.Row

        connection.execute("""
            CREATE TABLE IF NOT EXISTS tickets (
                transaction_id  TEXT PRIMARY KEY,
                created_at      REAL NOT NULL,
                status          TEXT NOT NULL,
                request_json    TEXT NOT NULL,
                result_json     TEXT,
                config_name     TEXT,
                state_key       TEXT,
                latency_seconds REAL,
                feedback_score  INTEGER,
                reward          REAL
            )
            """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS stage_runs (
                transaction_id TEXT NOT NULL,
                stage_name     TEXT NOT NULL,
                status         TEXT NOT NULL,
                output_json    TEXT,
                error          TEXT,
                started_at     REAL,
                finished_at    REAL,
                PRIMARY KEY (transaction_id, stage_name)
            )
            """)
        connection.commit()
        return connection

    # ------------------------------------------------------------------
    # Tickets
    # ------------------------------------------------------------------
    def create_ticket(self, transaction_id: str, request: dict) -> None:
        """Insert the ticket row before the pipeline starts."""
        with self.lock:
            connection = self._connect()
            try:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO tickets
                        (transaction_id, created_at, status, request_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (transaction_id, time.time(), STATUS_RUNNING, json.dumps(request)),
                )
                connection.commit()
            finally:
                connection.close()

    def finish_ticket(
        self,
        transaction_id: str,
        status: str,
        result: dict,
        config_name: str,
        state_key: str,
        latency_seconds: float,
    ) -> None:
        """Store the finished pipeline result and everything the RL loop needs later."""
        with self.lock:
            connection = self._connect()
            try:
                connection.execute(
                    """
                    UPDATE tickets
                       SET status = ?, result_json = ?, config_name = ?,
                           state_key = ?, latency_seconds = ?
                     WHERE transaction_id = ?
                    """,
                    (
                        status,
                        json.dumps(result),
                        config_name,
                        state_key,
                        latency_seconds,
                        transaction_id,
                    ),
                )
                connection.commit()
            finally:
                connection.close()

    def record_feedback(self, transaction_id: str, feedback_score: int, reward: float) -> None:
        """Store the feedback and the reward it produced."""
        with self.lock:
            connection = self._connect()
            try:
                connection.execute(
                    "UPDATE tickets SET feedback_score = ?, reward = ? WHERE transaction_id = ?",
                    (feedback_score, reward, transaction_id),
                )
                connection.commit()
            finally:
                connection.close()

    def get_ticket(self, transaction_id: str) -> dict | None:
        """Return the ticket row as a dictionary, or None if it does not exist."""
        with self.lock:
            connection = self._connect()
            try:
                cursor = connection.execute(
                    "SELECT * FROM tickets WHERE transaction_id = ?", (transaction_id,)
                )
                row = cursor.fetchone()
            finally:
                connection.close()

        if row is None:
            return None

        return {
            "transaction_id": row["transaction_id"],
            "created_at": row["created_at"],
            "status": row["status"],
            "request": json.loads(row["request_json"]),
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "config_name": row["config_name"],
            "state_key": row["state_key"],
            "latency_seconds": row["latency_seconds"],
            "feedback_score": row["feedback_score"],
            "reward": row["reward"],
        }

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------
    def start_stage(self, transaction_id: str, stage_name: str) -> None:
        with self.lock:
            connection = self._connect()
            try:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO stage_runs
                        (transaction_id, stage_name, status, output_json, error,
                         started_at, finished_at)
                    VALUES (?, ?, ?, NULL, NULL, ?, NULL)
                    """,
                    (transaction_id, stage_name, STATUS_RUNNING, time.time()),
                )
                connection.commit()
            finally:
                connection.close()

    def complete_stage(self, transaction_id: str, stage_name: str, output: dict) -> None:
        self._finish_stage(transaction_id, stage_name, STATUS_COMPLETED, output, "")

    def fail_stage(self, transaction_id: str, stage_name: str, error: str) -> None:
        self._finish_stage(transaction_id, stage_name, STATUS_FAILED, {}, error)

    def skip_stage(self, transaction_id: str, stage_name: str, reason: str) -> None:
        self._finish_stage(transaction_id, stage_name, STATUS_SKIPPED, {}, reason)

    def _finish_stage(
        self, transaction_id: str, stage_name: str, status: str, output: dict, error: str
    ) -> None:
        with self.lock:
            connection = self._connect()
            try:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO stage_runs
                        (transaction_id, stage_name, status, output_json, error,
                         started_at, finished_at)
                    VALUES (
                        ?, ?, ?, ?, ?,
                        COALESCE(
                            (SELECT started_at FROM stage_runs
                              WHERE transaction_id = ? AND stage_name = ?),
                            ?
                        ),
                        ?
                    )
                    """,
                    (
                        transaction_id,
                        stage_name,
                        status,
                        json.dumps(output),
                        error,
                        transaction_id,
                        stage_name,
                        time.time(),
                        time.time(),
                    ),
                )
                connection.commit()
            finally:
                connection.close()

    def get_stage(self, transaction_id: str, stage_name: str) -> StageRecord | None:
        with self.lock:
            connection = self._connect()
            try:
                cursor = connection.execute(
                    "SELECT * FROM stage_runs WHERE transaction_id = ? AND stage_name = ?",
                    (transaction_id, stage_name),
                )
                row = cursor.fetchone()
            finally:
                connection.close()

        if row is None:
            return None
        return _row_to_stage_record(row)

    def get_stages(self, transaction_id: str) -> list[StageRecord]:
        """Every stage row for one ticket, oldest first."""
        with self.lock:
            connection = self._connect()
            try:
                cursor = connection.execute(
                    """
                    SELECT * FROM stage_runs
                     WHERE transaction_id = ?
                     ORDER BY started_at ASC, stage_name ASC
                    """,
                    (transaction_id,),
                )
                rows = cursor.fetchall()
            finally:
                connection.close()

        records = []
        for row in rows:
            records.append(_row_to_stage_record(row))
        return records

    def clear_stage(self, transaction_id: str, stage_name: str) -> None:
        """Forget one stage so that a re-run will execute it again."""
        with self.lock:
            connection = self._connect()
            try:
                connection.execute(
                    "DELETE FROM stage_runs WHERE transaction_id = ? AND stage_name = ?",
                    (transaction_id, stage_name),
                )
                connection.commit()
            finally:
                connection.close()


def _row_to_stage_record(row: sqlite3.Row) -> StageRecord:
    """Turn a database row into a StageRecord."""
    output: dict = {}
    if row["output_json"]:
        output = json.loads(row["output_json"])

    return StageRecord(
        stage_name=row["stage_name"],
        status=row["status"],
        output=output,
        error=row["error"] or "",
        started_at=row["started_at"] or 0.0,
        finished_at=row["finished_at"] or 0.0,
    )
