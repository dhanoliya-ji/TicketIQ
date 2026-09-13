/* ===========================================================================
   TicketIQ console.

   Plain browser JavaScript, no framework and no build step. It talks to the
   same FastAPI service that serves this page, so every value shown comes from
   a real API call.

   The page deliberately shows the reply first and keeps the supporting
   evidence - sentiment, sources, reasoning, timings, routing - in collapsed
   sections, so the common case is a short page and the detail is one click
   away when someone asks "why did it say that?".
   =========================================================================== */

"use strict";

/* ---------------------------------------------------------------------------
   Examples. These exist so a demo does not start with typing.
   --------------------------------------------------------------------------- */
var EXAMPLES = [
  {
    label: "Duplicate charge",
    subject: "Charged twice on order 4471",
    body:
      "My credit card was charged twice for the same monthly invoice. I want a " +
      "refund and nobody has replied for days.",
    tier: "enterprise",
  },
  {
    label: "Outage",
    subject: "Total outage, dashboard is down",
    body:
      "Every API call returns a 500 error and the whole platform is unusable. " +
      "Our team is completely blocked.",
    tier: "enterprise",
  },
  {
    label: "Login problem",
    subject: "Cannot log in to my account",
    body:
      "The password reset email never arrives for customer 3391 and I am locked " +
      "out of the workspace.",
    tier: "pro",
  },
  {
    label: "Feature request",
    subject: "Please add dark mode",
    body: "It would be great if the product supported a dark theme for night work.",
    tier: "free",
  },
];

// The ticket currently on screen.
var current = null;
var isBusy = false;

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

  if (detail && typeof detail === "string") {
    return detail;
  }
  if (detail && typeof detail === "object" && detail.message) {
    var text = detail.message;
    if (detail.failed_stage) {
      text = text + " (stage: " + detail.failed_stage + ")";
    }
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
  byId("submit-label").textContent = busy ? "Working…" : "Triage ticket";
}

function showError(message) {
  var element = byId("form-error");
  if (!message) {
    element.hidden = true;
    return;
  }
  element.textContent = message;
  element.hidden = false;
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
    loadRouting(result.rl_state_key);

    byId("result").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showError(error.message);
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

    var head = make("div", "aspect-head");
    head.appendChild(make("span", "aspect-name", titleCase(aspect.aspect)));

    var label = make("span", "aspect-label", aspect.label);
    label.dataset.polarity = aspect.label;
    head.appendChild(label);

    head.appendChild(make("span", "aspect-score", aspect.score.toFixed(2)));
    item.appendChild(head);

    if (aspect.evidence) {
      item.appendChild(make("div", "aspect-quote", "“" + aspect.evidence + "”"));
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

    var head = make("div", "source-head");
    head.appendChild(make("span", "source-heading", snippet.heading));
    head.appendChild(make("span", "source-file", snippet.source));
    item.appendChild(head);

    item.appendChild(make("div", "source-text", snippet.text.replace(/\n/g, " ")));
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
      row.appendChild(make("td", null, stage.stage));
      row.appendChild(
        make("td", stage.status === "completed" ? "cell-good" : "cell-critical", stage.status)
      );
      row.appendChild(make("td", null, (stage.duration_seconds * 1000).toFixed(1) + " ms"));
      tableBody.appendChild(row);
    });

    byId("hint-processing").textContent =
      status.completed_stages + " of " + status.total_stages + " stages completed";
  } catch (error) {
    byId("hint-processing").textContent = "unavailable";
  }
}

/* ---------------------------------------------------------------------------
   Routing performance: what the bandit has learned for this kind of ticket
   --------------------------------------------------------------------------- */
async function loadRouting(stateKey) {
  var tableBody = byId("out-routing").querySelector("tbody");
  clear(tableBody);

  try {
    var stats = await apiGet("/rl/stats");
    var arms = stats.states[stateKey];

    byId("routing-note").textContent =
      "Reply style and how much policy to retrieve are chosen by an online " +
      "bandit, scored as (feedback × 10) − latency. Figures below are for " +
      "tickets like this one (" +
      stateKey.split("|").join(", ") +
      ").";

    if (!arms) {
      byId("hint-routing").textContent = "no data yet";
      var empty = make("tr");
      var cell = make("td", null, "No feedback recorded for this ticket type yet.");
      cell.colSpan = 3;
      empty.appendChild(cell);
      tableBody.appendChild(empty);
      return;
    }

    var bestName = null;
    var bestReward = -Infinity;
    var totalPulls = 0;

    stats.actions.forEach(function (name) {
      totalPulls = totalPulls + arms[name].pulls;
      if (arms[name].pulls > 0 && arms[name].average_reward > bestReward) {
        bestReward = arms[name].average_reward;
        bestName = name;
      }
    });

    stats.actions.forEach(function (name) {
      var arm = arms[name];
      var row = make("tr");
      if (name === bestName) {
        row.dataset.best = "true";
      }

      row.appendChild(make("td", null, name));
      row.appendChild(make("td", null, arm.pulls));
      row.appendChild(
        make("td", null, arm.pulls === 0 ? "—" : arm.average_reward.toFixed(2))
      );
      tableBody.appendChild(row);
    });

    byId("hint-routing").textContent =
      totalPulls === 0
        ? "no data yet"
        : totalPulls + " rating" + (totalPulls === 1 ? "" : "s") + " so far";
  } catch (error) {
    byId("hint-routing").textContent = "unavailable";
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
    note.appendChild(document.createTextNode("Thank you. Recorded a score of "));
    note.appendChild(make("strong", null, response.reward.toFixed(2)));
    note.appendChild(
      document.createTextNode(
        " for this routing choice, and the preferred setup for tickets like this " +
          "is now " + response.best_config_for_state + "."
      )
    );
    note.hidden = false;

    loadRouting(response.state_key);
  } catch (error) {
    note.textContent = error.message;
    note.hidden = false;
  }
}

/* ---------------------------------------------------------------------------
   Start-up
   --------------------------------------------------------------------------- */
function buildExamples() {
  var container = byId("examples");

  EXAMPLES.forEach(function (example, index) {
    if (index > 0) {
      // A visible separator: without it the underlined labels run together.
      container.appendChild(make("span", "example-separator", "·"));
    }

    var button = make("button", "example-link", example.label);
    button.type = "button";
    button.addEventListener("click", function () {
      byId("subject").value = example.subject;
      byId("body").value = example.body;
      byId("tier").value = example.tier;
      showError(null);
    });
    container.appendChild(button);
  });
}

async function loadStatus() {
  var dot = byId("status-dot");
  var text = byId("status-text");

  try {
    var health = await apiGet("/health");
    dot.className = "status-dot is-ready";
    text.textContent = health.ready ? "Ready" : "Starting";
  } catch (error) {
    dot.className = "status-dot is-down";
    text.textContent = "Service unavailable";
  }
}

function start() {
  buildExamples();

  byId("ticket-form").addEventListener("submit", submitTicket);
  byId("fb-yes").addEventListener("click", function () {
    sendFeedback(1);
  });
  byId("fb-no").addEventListener("click", function () {
    sendFeedback(0);
  });

  loadStatus();

  // Start on the first example so the page is one click from a result.
  byId("subject").value = EXAMPLES[0].subject;
  byId("body").value = EXAMPLES[0].body;
  byId("tier").value = EXAMPLES[0].tier;
}

document.addEventListener("DOMContentLoaded", start);
