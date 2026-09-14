/* ===========================================================================
   TicketIQ console.

   Plain browser JavaScript, no framework and no build step. It talks to the
   same FastAPI service that serves this page, so every value shown comes from
   a real API call and nothing is computed here.

   The page is exactly the three endpoints the assignment specifies: POST
   /ticket takes a free-text subject, body and customer tier; the result shows
   everything that endpoint returns; GET /ticket/{id}/status supplies the stage
   table; POST /feedback records the rating.
   =========================================================================== */

"use strict";

// The ticket currently on screen.
var current = null;
var isBusy = false;

// Set when a pipeline failure came back with a transaction id, so the error
// message can offer to retry that transaction.
var lastFailedTransactionId = null;

/* ---------------------------------------------------------------------------
   Small helpers
   --------------------------------------------------------------------------- */
function byId(id) {
  return document.getElementById(id);
}

function make(tag, className, text) {
  var element = document.createElement(tag);
  if (className) {
    element.className = className;
  }
  if (text !== undefined && text !== null) {
    element.textContent = String(text);
  }
  return element;
}

function clear(element) {
  while (element.firstChild) {
    element.removeChild(element.firstChild);
  }
}

function titleCase(value) {
  var spaced = String(value).replace(/_/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

function seconds(value) {
  return Number(value).toFixed(2) + " s";
}

/* ---------------------------------------------------------------------------
   API
   --------------------------------------------------------------------------- */
async function apiGet(path) {
  var response = await fetch(path);
  if (!response.ok) {
    throw new Error(describeFailure(response.status, null));
  }
  return response.json();
}

async function apiPost(path, payload) {
  var response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  var body = null;
  try {
    body = await response.json();
  } catch (error) {
    body = null;
  }

  if (!response.ok) {
    throw new Error(describeFailure(response.status, body));
  }
  return body;
}

/* Turn a failed response into something a person can act on, rather than
   showing a bare status code. */
function describeFailure(status, body) {
  var detail = body ? body.detail : null;
  lastFailedTransactionId = null;

  if (detail && typeof detail === "string") {
    return detail;
  }
  if (detail && typeof detail === "object" && detail.message) {
    var text = detail.message;
    if (detail.failed_stage) {
      text = text + " (stage: " + detail.failed_stage + ")";
    }
    // Remembered so the caller can offer a retry of just that stage.
    lastFailedTransactionId = detail.transaction_id || null;
    return text;
  }
  if (Array.isArray(detail) && detail.length > 0 && detail[0].msg) {
    return detail[0].msg;
  }

  if (status === 404) {
    return "That ticket could not be found.";
  }
  if (status === 409) {
    return "Feedback has already been recorded for this ticket.";
  }
  if (status >= 500) {
    return "The service failed to process this request. Check the server log for details.";
  }
  return "Request failed with status " + status + ".";
}

/* ---------------------------------------------------------------------------
   Submitting a ticket
   --------------------------------------------------------------------------- */
function setBusy(busy) {
  isBusy = busy;
  var button = byId("submit-button");
  button.disabled = busy;
  button.classList.toggle("is-busy", busy);
  byId("submit-label").textContent = busy ? "Working…" : "Submit ticket";
}

/* Show an error. When the pipeline failed part way, the service gives back the
   transaction id, so we can offer to retry it - which re-runs only the broken
   stage rather than the whole pipeline. */
function showError(message, retryTransactionId) {
  var element = byId("form-error");
  clear(element);

  if (!message) {
    element.hidden = true;
    return;
  }

  element.appendChild(document.createTextNode(message));

  if (retryTransactionId) {
    var button = make("button", "button retry-button", "Retry the failed step");
    button.type = "button";
    button.addEventListener("click", function () {
      retryTicket(retryTransactionId, button);
    });
    element.appendChild(button);
  }

  element.hidden = false;
}

/* Re-run a failed ticket. The stages that already succeeded are reused. */
async function retryTicket(transactionId, button) {
  button.disabled = true;
  button.textContent = "Retrying…";

  try {
    var result = await apiPost("/ticket/" + transactionId + "/retry", {});
    current = result;
    showError(null);
    render(result);
    loadStages(result.transaction_id);
    byId("result").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    button.disabled = false;
    button.textContent = "Retry the failed step";
    showError(error.message, transactionId);
  }
}

async function submitTicket(event) {
  event.preventDefault();
  if (isBusy) {
    return;
  }

  var subject = byId("subject").value.trim();
  var body = byId("body").value.trim();

  if (subject === "" || body === "") {
    showError("Please fill in both a subject and a description.");
    return;
  }

  showError(null);
  setBusy(true);

  // Hide the previous result straight away. Leaving it on screen while the new
  // one is being worked out makes stale figures look like fresh ones.
  byId("result").hidden = true;
  current = null;

  try {
    var result = await apiPost("/ticket", {
      subject: subject,
      body: body,
      customer_tier: byId("tier").value,
    });

    current = result;
    render(result);

    // The per-stage timings come from the workflow engine's own state store.
    loadStages(result.transaction_id);

    byId("result").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showError(error.message, lastFailedTransactionId);
  } finally {
    setBusy(false);
  }
}

/* ---------------------------------------------------------------------------
   Rendering the result
   --------------------------------------------------------------------------- */
function render(result) {
  byId("result").hidden = false;
  byId("result-ref").textContent = result.transaction_id;

  byId("out-category").textContent = titleCase(result.category);
  byId("out-latency").textContent = seconds(result.latency_seconds);

  var urgency = byId("out-urgency");
  urgency.textContent = titleCase(result.urgency_bucket);
  urgency.dataset.level = result.urgency_bucket;

  var action = byId("out-action");
  var escalated = result.action === "escalate_to_human";
  action.textContent = escalated ? "Escalate to human" : "Answer sent";
  action.dataset.level = escalated ? "escalate" : "answer";

  // The assignment asks for the configuration used, named as
  // model / prompt variant / RAG top-K.
  var config = result.pipeline_config;
  byId("out-config").textContent = config.prompt_variant;
  byId("out-config-detail").textContent =
    config.model + " · top-K " + config.rag_top_k;

  byId("out-reply").textContent = result.response_text;

  renderAspects(result.aspect_sentiments);
  renderSources(result.retrieved_knowledge);
  renderTrace(result.reasoning_trace);

  byId("hint-analysis").textContent =
    result.aspect_sentiments.length +
    " aspect" +
    (result.aspect_sentiments.length === 1 ? "" : "s") +
    ", " +
    result.retrieved_knowledge.length +
    " source" +
    (result.retrieved_knowledge.length === 1 ? "" : "s");

  var toolCount = result.tool_calls.length;
  byId("hint-reasoning").textContent =
    result.reasoning_trace.length +
    " step" +
    (result.reasoning_trace.length === 1 ? "" : "s") +
    (toolCount > 0 ? ", " + toolCount + " tool call" + (toolCount === 1 ? "" : "s") : "");

  // Reset the feedback controls for this new ticket.
  byId("fb-yes").disabled = false;
  byId("fb-no").disabled = false;
  byId("feedback-note").hidden = true;
}

function renderAspects(aspects) {
  var list = byId("out-aspects");
  clear(list);

  aspects.forEach(function (aspect) {
    var item = make("li");

    var head = make("div", "row-head");
    head.appendChild(make("span", "row-name", titleCase(aspect.aspect)));

    var polarity = make("span", "polarity", aspect.label);
    polarity.dataset.polarity = aspect.label;
    head.appendChild(polarity);

    head.appendChild(make("span", "score", aspect.score.toFixed(2)));
    item.appendChild(head);

    if (aspect.evidence) {
      item.appendChild(make("div", "row-quote", "“" + aspect.evidence + "”"));
    }
    list.appendChild(item);
  });
}

function renderSources(snippets) {
  var list = byId("out-sources");
  clear(list);

  if (snippets.length === 0) {
    list.appendChild(make("li", null, "No knowledge base section matched this ticket."));
    return;
  }

  snippets.forEach(function (snippet) {
    var item = make("li");

    var head = make("div", "row-head");
    head.appendChild(make("span", "row-name", snippet.heading));
    head.appendChild(make("span", "row-file", snippet.source));
    item.appendChild(head);

    item.appendChild(make("div", "row-body", snippet.text.replace(/\s+/g, " ")));
    list.appendChild(item);
  });
}

function renderTrace(steps) {
  var list = byId("out-trace");
  clear(list);

  steps.forEach(function (step) {
    var item = make("li");
    var terminal = step.action === "answer" || step.action === "escalate_to_human";
    item.dataset.terminal = terminal ? "true" : "false";

    item.appendChild(make("div", "trace-action", step.step + ". " + step.action));
    item.appendChild(make("div", "trace-thought", step.thought));

    if (step.observation) {
      item.appendChild(make("div", "trace-observation", step.observation));
    }
    // Shown when a policy rule changed what the model asked for, so the
    // override is visible rather than silently applied.
    if (step.override) {
      item.appendChild(make("div", "trace-override", "Adjusted — " + step.override));
    }
    list.appendChild(item);
  });
}

/* ---------------------------------------------------------------------------
   Processing steps, read from the workflow engine's state store
   --------------------------------------------------------------------------- */
async function loadStages(transactionId) {
  var tableBody = byId("out-stages").querySelector("tbody");
  clear(tableBody);
  byId("hint-processing").textContent = "";

  try {
    var status = await apiGet("/ticket/" + transactionId + "/status");

    status.stages.forEach(function (stage) {
      var row = make("tr");
      row.appendChild(make("td", "name", stage.stage));
      row.appendChild(
        make("td", stage.status === "completed" ? "cell-good" : "cell-critical", stage.status)
      );
      row.appendChild(
        make("td", "numeric", (stage.duration_seconds * 1000).toFixed(1) + " ms")
      );
      tableBody.appendChild(row);
    });

    byId("hint-processing").textContent =
      status.completed_stages + " of " + status.total_stages + " stages completed";
  } catch (error) {
    byId("hint-processing").textContent = "unavailable";
  }
}

/* ---------------------------------------------------------------------------
   Feedback
   --------------------------------------------------------------------------- */
async function sendFeedback(score) {
  if (!current) {
    return;
  }

  byId("fb-yes").disabled = true;
  byId("fb-no").disabled = true;

  var note = byId("feedback-note");

  try {
    var response = await apiPost("/feedback", {
      transaction_id: current.transaction_id,
      feedback_score: score,
    });

    clear(note);
    note.appendChild(document.createTextNode("Recorded. Reward "));
    note.appendChild(make("strong", null, response.reward.toFixed(2)));
    note.appendChild(
      document.createTextNode(
        " — feedback × 10 minus the " +
          response.latency_seconds.toFixed(2) + "s it took. The best configuration " +
          "for this kind of ticket is now " + response.best_config_for_state + "."
      )
    );
    note.hidden = false;
  } catch (error) {
    note.textContent = error.message;
    note.hidden = false;
  }
}

/* ---------------------------------------------------------------------------
   Start-up
   --------------------------------------------------------------------------- */
function start() {
  byId("ticket-form").addEventListener("submit", submitTicket);
  byId("fb-yes").addEventListener("click", function () {
    sendFeedback(1);
  });
  byId("fb-no").addEventListener("click", function () {
    sendFeedback(0);
  });

}

document.addEventListener("DOMContentLoaded", start);
