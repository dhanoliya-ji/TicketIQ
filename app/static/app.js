/* ===========================================================================
   TicketIQ live console.

   Plain browser JavaScript, no framework and no build step. It talks to the
   same FastAPI service that serves this page, so every number on screen comes
   from a real API call - nothing here is faked or pre-computed.

   Reading order:
     1. constants and small helpers
     2. API calls
     3. the workflow DAG animation
     4. rendering one triage result
     5. the reinforcement learning dashboard
     6. the guided "watch it learn" demo
     7. start-up wiring
   =========================================================================== */

"use strict";

/* ---------------------------------------------------------------------------
   1. Constants and helpers
   --------------------------------------------------------------------------- */

// The four bandit arms, in a fixed order with a fixed colour each. Colour must
// follow the configuration, never its current rank, so a bar never changes hue
// when the ordering changes.
var CONFIG_COLORS = {
  "concise_policy|k2": "var(--series-1)",
  "concise_policy|k5": "var(--series-2)",
  "empathetic_stepwise|k2": "var(--series-3)",
  "empathetic_stepwise|k5": "var(--series-4)",
};

// Each ticket category gets a hue, but the word is always shown next to it.
var CATEGORY_COLORS = {
  billing: "var(--series-1)",
  technical: "var(--series-2)",
  account: "var(--series-3)",
  feature_request: "var(--series-4)",
};

// Sentiment and urgency are status colours: always paired with their label.
var SENTIMENT_COLORS = {
  positive: "var(--status-good)",
  neutral: "var(--text-muted)",
  negative: "var(--status-critical)",
};

var URGENCY_COLORS = {
  low: "var(--status-good)",
  medium: "var(--status-warning)",
  high: "var(--status-critical)",
};

// One-click sample tickets, chosen to show four different agent behaviours.
var SAMPLE_TICKETS = [
  {
    emoji: "💳",
    label: "Duplicate charge",
    hue: "var(--series-1)",
    subject: "Charged twice on order 4471",
    body:
      "My credit card was charged twice for the same monthly invoice. I want a " +
      "refund and nobody has replied for days.",
    tier: "enterprise",
  },
  {
    emoji: "🔥",
    label: "Enterprise outage",
    hue: "var(--series-2)",
    subject: "Total outage, dashboard is down",
    body:
      "Every API call returns a 500 error and the whole platform is unusable. " +
      "Our team is completely blocked and this is costing us money.",
    tier: "enterprise",
  },
  {
    emoji: "🔑",
    label: "Locked out",
    hue: "var(--series-3)",
    subject: "Cannot log in to my account",
    body:
      "The password reset email never arrives for customer 3391 and I am locked " +
      "out of the workspace.",
    tier: "pro",
  },
  {
    emoji: "🌙",
    label: "Feature request",
    hue: "var(--series-4)",
    subject: "Please add dark mode",
    body: "It would be great if the product supported a dark theme for night work.",
    tier: "free",
  },
];

// Short human descriptions of each stage, shown inside the DAG nodes.
var STAGE_DESCRIPTIONS = {
  classify_ticket: "Naive Bayes category",
  analyse_sentiment: "VADER, per aspect",
  score_urgency: "category + sentiment + tier",
  select_configuration: "contextual bandit picks an arm",
  retrieve_knowledge: "cosine search, top-K",
  run_agent: "ReAct loop and tools",
  compose_response: "final customer reply",
};

// The pipeline finishes in about a tenth of a second, which is far too fast to
// watch. The DAG replays the *real* recorded stage durations, slowed by this
// factor so a person can follow the order. The note under the heading says so.
var REPLAY_SLOWDOWN = 26;
var REPLAY_MINIMUM_MS = 180;
var REPLAY_MAXIMUM_MS = 900;

// Module state: the ticket currently on screen.
var currentTransactionId = null;
var currentStateKey = null;
var isBusy = false;

function byId(id) {
  return document.getElementById(id);
}

function createElement(tagName, className, text) {
  var element = document.createElement(tagName);
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

function wait(milliseconds) {
  return new Promise(function (resolve) {
    setTimeout(resolve, milliseconds);
  });
}

function formatSeconds(value) {
  return Number(value).toFixed(3) + " s";
}

function formatMilliseconds(seconds) {
  return (Number(seconds) * 1000).toFixed(1) + " ms";
}

/* Counts a number up, because a value that animates draws the eye to the fact
   that it changed. */
function countUp(element, target) {
  var start = Number(element.textContent.replace(/[^\d.-]/g, "")) || 0;
  var end = Number(target);
  if (start === end) {
    return;
  }

  var steps = 18;
  var step = 0;
  var timer = setInterval(function () {
    step = step + 1;
    var value = start + ((end - start) * step) / steps;
    element.textContent = String(Math.round(value));
    if (step >= steps) {
      clearInterval(timer);
      element.textContent = String(end);
    }
  }, 22);
}

/* ---------------------------------------------------------------------------
   2. API calls, each one logged to the activity panel
   --------------------------------------------------------------------------- */

function logRequest(method, path, ok, detail) {
  var list = byId("log");
  var empty = list.querySelector(".log-empty");
  if (empty) {
    list.removeChild(empty);
  }

  var row = createElement("li");
  var now = new Date();
  row.appendChild(createElement("span", "log-time", now.toLocaleTimeString()));
  row.appendChild(createElement("span", "log-method " + (ok ? "log-ok" : "log-fail"), method));
  row.appendChild(createElement("span", null, path));
  if (detail) {
    row.appendChild(createElement("span", "log-detail", detail));
  }

  list.insertBefore(row, list.firstChild);

  // Keep the panel short; old rows are not interesting.
  while (list.children.length > 40) {
    list.removeChild(list.lastChild);
  }
}

async function apiGet(path, quiet) {
  var response = await fetch(path);
  if (!quiet) {
    logRequest("GET", path, response.ok, response.status + "");
  }
  if (!response.ok) {
    throw new Error("GET " + path + " failed with status " + response.status);
  }
  return response.json();
}

async function apiPost(path, payload, quiet) {
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

  if (!quiet) {
    logRequest("POST", path, response.ok, response.status + "");
  }

  if (!response.ok) {
    var message = "request failed with status " + response.status;
    if (body && body.detail) {
      message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    }
    throw new Error(message);
  }
  return body;
}

/* ---------------------------------------------------------------------------
   3. The workflow DAG
   --------------------------------------------------------------------------- */

/* Draw the graph from /workflow/graph, so the picture is the service's real
   dependency structure rather than a drawing that could drift out of date. */
async function buildDag() {
  var graph = await apiGet("/workflow/graph", true);
  var container = byId("dag");
  clear(container);

  graph.levels.forEach(function (level, index) {
    if (index > 0) {
      container.appendChild(createElement("div", "dag-arrow", "▼"));
    }

    var row = createElement("div", "dag-level");
    row.dataset.parallel = level.runs_in_parallel ? "true" : "false";

    level.stages.forEach(function (stage) {
      var node = createElement("div", "dag-node");
      node.dataset.stage = stage.name;

      node.appendChild(createElement("span", "node-icon", String(index)));

      var textWrap = createElement("div");
      textWrap.appendChild(createElement("div", "node-name", stage.name));
      textWrap.appendChild(
        createElement("div", "node-desc", STAGE_DESCRIPTIONS[stage.name] || stage.description)
      );
      node.appendChild(textWrap);

      node.appendChild(createElement("span", "node-time", ""));
      row.appendChild(node);
    });

    if (level.runs_in_parallel) {
      // The tag belongs to the level, not to either stage in it.
      row.appendChild(createElement("span", "parallel-tag", "these two run in parallel"));
    }

    container.appendChild(row);
  });
}

function resetDag() {
  var nodes = document.querySelectorAll(".dag-node");
  for (var i = 0; i < nodes.length; i = i + 1) {
    nodes[i].classList.remove("is-running", "is-done", "is-failed");
    nodes[i].querySelector(".node-icon").textContent = nodes[i]
      .closest(".dag-level")
      .previousElementSibling
      ? nodes[i].querySelector(".node-icon").textContent
      : nodes[i].querySelector(".node-icon").textContent;
    nodes[i].querySelector(".node-time").textContent = "";
  }
  byId("replay-note").hidden = true;
}

function markStage(stageName, state, durationSeconds) {
  var node = document.querySelector('.dag-node[data-stage="' + stageName + '"]');
  if (!node) {
    return;
  }

  node.classList.remove("is-running", "is-done", "is-failed");
  node.classList.add("is-" + state);

  if (state === "done") {
    node.querySelector(".node-icon").textContent = "✓";
    node.querySelector(".node-time").textContent = formatMilliseconds(durationSeconds);
  } else if (state === "failed") {
    node.querySelector(".node-icon").textContent = "!";
  }
}

/* Replay the recorded stage timings. Stages on the same level are started at
   the same moment, which is how they actually ran. */
async function replayDag(stageRecords) {
  var byLevel = [];
  var levels = document.querySelectorAll(".dag-level");

  for (var i = 0; i < levels.length; i = i + 1) {
    var names = [];
    var nodes = levels[i].querySelectorAll(".dag-node");
    for (var j = 0; j < nodes.length; j = j + 1) {
      names.push(nodes[j].dataset.stage);
    }
    byLevel.push(names);
  }

  var durations = {};
  stageRecords.forEach(function (record) {
    durations[record.stage] = record.duration_seconds;
  });

  for (var level = 0; level < byLevel.length; level = level + 1) {
    var names = byLevel[level];
    var slowest = 0;

    names.forEach(function (name) {
      markStage(name, "running");
      var realMs = (durations[name] || 0) * 1000 * REPLAY_SLOWDOWN;
      var shownMs = Math.min(REPLAY_MAXIMUM_MS, Math.max(REPLAY_MINIMUM_MS, realMs));
      if (shownMs > slowest) {
        slowest = shownMs;
      }
    });

    await wait(slowest);

    names.forEach(function (name) {
      markStage(name, "done", durations[name] || 0);
    });
  }
}

/* ---------------------------------------------------------------------------
   4. Rendering one triage result
   --------------------------------------------------------------------------- */

function renderResult(result) {
  byId("result-panel").hidden = false;
  byId("result-tx").textContent = result.transaction_id;

  // --- category and confidence ---
  var categoryElement = byId("out-category");
  categoryElement.textContent = result.category;
  categoryElement.style.color = CATEGORY_COLORS[result.category] || "var(--text-primary)";

  var confidencePercent = Math.round(result.category_confidence * 100);
  byId("out-confidence").textContent = confidencePercent + "%";
  // The width is set on the next frame so the transition actually runs.
  requestAnimationFrame(function () {
    byId("out-confidence-bar").style.width = confidencePercent + "%";
  });

  // --- urgency ---
  var bucket = result.urgency_bucket;
  var urgencyColor = URGENCY_COLORS[bucket] || "var(--text-muted)";
  var bucketElement = byId("out-urgency-bucket");
  bucketElement.textContent = bucket;
  bucketElement.style.color = urgencyColor;
  byId("out-urgency-score").textContent = result.urgency_score.toFixed(3);

  var urgencyBar = byId("out-urgency-bar");
  urgencyBar.style.background = urgencyColor;
  requestAnimationFrame(function () {
    urgencyBar.style.width = Math.round(result.urgency_score * 100) + "%";
  });

  // --- agent decision ---
  var actionElement = byId("out-action");
  actionElement.textContent = result.action.replace(/_/g, " ");
  actionElement.dataset.action = result.action;

  var toolNames = result.tool_calls.map(function (call) {
    return call.tool;
  });
  byId("out-tools").textContent =
    toolNames.length > 0 ? "tools called: " + toolNames.join(", ") : "no tools were needed";

  // --- configuration chosen by the bandit ---
  byId("out-config").textContent = result.pipeline_config.name;
  byId("out-config").style.color = CONFIG_COLORS[result.pipeline_config.name];
  byId("out-reason").textContent = result.config_selection_reason.replace(/_/g, " ");
  byId("out-state").textContent = result.rl_state_key;

  renderAspects(result.aspect_sentiments);
  renderSnippets(result.retrieved_knowledge, result.pipeline_config.rag_top_k);
  renderTrace(result.reasoning_trace);

  byId("out-reply").textContent = result.response_text;

  // --- reset the feedback controls for this new ticket ---
  currentTransactionId = result.transaction_id;
  currentStateKey = result.rl_state_key;
  byId("fb-up").disabled = false;
  byId("fb-down").disabled = false;
  byId("feedback-result").hidden = true;
}

function renderAspects(aspects) {
  var container = byId("out-aspects");
  clear(container);

  aspects.forEach(function (aspect, index) {
    var color = SENTIMENT_COLORS[aspect.label] || "var(--text-muted)";

    var card = createElement("div", "aspect");
    card.style.setProperty("--aspect-hue", color);
    card.style.animationDelay = index * 60 + "ms";

    card.appendChild(createElement("div", "aspect-name", aspect.aspect.replace(/_/g, " ")));

    var meta = createElement("div", "aspect-meta");
    meta.appendChild(createElement("span", "aspect-label", aspect.label));
    meta.appendChild(createElement("span", "aspect-score", aspect.score.toFixed(3)));
    card.appendChild(meta);

    if (aspect.evidence) {
      card.appendChild(createElement("div", "aspect-evidence", "“" + aspect.evidence + "”"));
    }

    container.appendChild(card);
  });
}

function renderSnippets(snippets, topK) {
  byId("out-topk").textContent = "top-K = " + topK + ", " + snippets.length + " returned";

  var container = byId("out-snippets");
  clear(container);

  if (snippets.length === 0) {
    container.appendChild(
      createElement("p", "bars-empty", "No knowledge base section matched this ticket.")
    );
    return;
  }

  // Similarity is a magnitude, so the bars share one sequential hue and are
  // scaled against the best hit rather than against 1.0, which would make
  // every bar look tiny.
  var best = snippets[0].score || 1;

  snippets.forEach(function (snippet, index) {
    var card = createElement("div", "snippet");
    card.style.animationDelay = index * 60 + "ms";

    var head = createElement("div", "snippet-head");
    head.appendChild(createElement("span", "snippet-heading", snippet.heading));
    head.appendChild(createElement("span", "snippet-source", snippet.source));

    var score = createElement("div", "snippet-score");
    var track = createElement("div", "snippet-score-bar");
    var fill = createElement("div", "snippet-score-fill");
    track.appendChild(fill);
    score.appendChild(track);
    score.appendChild(createElement("span", "snippet-score-value", snippet.score.toFixed(3)));
    head.appendChild(score);

    card.appendChild(head);
    card.appendChild(createElement("div", "snippet-text", snippet.text.replace(/\n/g, " ")));
    container.appendChild(card);

    requestAnimationFrame(function () {
      fill.style.width = Math.round((snippet.score / best) * 100) + "%";
    });
  });
}

function renderTrace(steps) {
  var container = byId("out-trace");
  clear(container);

  steps.forEach(function (step, index) {
    var item = createElement("li", "trace-step");
    item.style.animationDelay = index * 80 + "ms";

    if (step.action === "escalate_to_human") {
      item.dataset.kind = "escalate";
    } else if (step.observation) {
      item.dataset.kind = "tool";
    } else {
      item.dataset.kind = "answer";
    }

    item.appendChild(
      createElement("div", "trace-action", "step " + step.step + " → " + step.action)
    );
    item.appendChild(createElement("div", "trace-thought", step.thought));

    if (step.observation) {
      item.appendChild(createElement("div", "trace-observation", "↳ " + step.observation));
    }
    container.appendChild(item);
  });
}

/* ---------------------------------------------------------------------------
   5. The reinforcement learning dashboard
   --------------------------------------------------------------------------- */

var latestStats = null;

/* The state with the most rewards applied - the one worth showing first. */
function busiestState(stateNames) {
  var bestName = stateNames[0];
  var bestPulls = -1;

  stateNames.forEach(function (name) {
    var arms = latestStats.states[name];
    var pulls = 0;
    Object.keys(arms).forEach(function (action) {
      pulls = pulls + arms[action].pulls;
    });
    if (pulls > bestPulls) {
      bestPulls = pulls;
      bestName = name;
    }
  });

  return bestName;
}

async function refreshRlDashboard(preferredState) {
  latestStats = await apiGet("/rl/stats", true);

  countUp(byId("tile-updates"), latestStats.total_updates);
  countUp(byId("tile-states"), Object.keys(latestStats.states).length);
  countUp(byId("tile-explore"), latestStats.exploration_count);
  countUp(byId("tile-exploit"), latestStats.exploitation_count);

  byId("pill-updates").querySelector(".pill-value").textContent = latestStats.total_updates;

  var stateNames = Object.keys(latestStats.states).sort();
  var select = byId("state-select");

  // Default to the state with the most feedback, so the chart opens on the
  // state that actually has something to show rather than on whichever name
  // happens to sort first.
  var chosen = preferredState || select.value || busiestState(stateNames);

  clear(select);
  stateNames.forEach(function (name) {
    var option = createElement("option", null, name);
    option.value = name;
    select.appendChild(option);
  });

  if (stateNames.indexOf(chosen) === -1) {
    chosen = stateNames[0];
  }
  if (chosen) {
    select.value = chosen;
  }

  renderBars(chosen);
}

function renderBars(stateKey) {
  var container = byId("rl-bars");
  var tableBody = byId("rl-table").querySelector("tbody");
  clear(container);
  clear(tableBody);

  if (!latestStats || !stateKey || !latestStats.states[stateKey]) {
    byId("chart-state-sub").textContent =
      "No feedback recorded yet — run a ticket and rate the answer.";
    container.appendChild(
      createElement("p", "bars-empty", "Nothing learned yet. Rate an answer to create the first bar.")
    );
    return;
  }

  var arms = latestStats.states[stateKey];
  var names = latestStats.actions;

  var totalPulls = 0;
  var bestName = null;
  var bestReward = -Infinity;
  var largestMagnitude = 0.001;

  names.forEach(function (name) {
    var arm = arms[name];
    totalPulls = totalPulls + arm.pulls;
    if (arm.pulls > 0 && arm.average_reward > bestReward) {
      bestReward = arm.average_reward;
      bestName = name;
    }
    largestMagnitude = Math.max(largestMagnitude, Math.abs(arm.average_reward));
  });

  byId("chart-state-sub").textContent =
    "State " + stateKey + " · " + totalPulls + " reward" + (totalPulls === 1 ? "" : "s") + " applied";

  names.forEach(function (name) {
    var arm = arms[name];
    var color = CONFIG_COLORS[name];

    var row = createElement("div", "bar-row");
    if (name === bestName) {
      row.classList.add("is-best");
    }

    // Name, with its colour swatch. The swatch alone never carries the
    // identity - the configuration name is written out beside it.
    var nameCell = createElement("div", "bar-name");
    var swatch = createElement("span", "bar-swatch");
    swatch.style.background = color;
    nameCell.appendChild(swatch);
    nameCell.appendChild(createElement("span", null, name));
    row.appendChild(nameCell);

    var track = createElement("div", "bar-track");
    var fill = createElement("div", "bar-fill");
    fill.style.background = color;
    if (arm.average_reward < 0) {
      fill.classList.add("is-negative");
    }
    track.appendChild(fill);
    row.appendChild(track);

    var value = createElement("div", "bar-value");
    var strong = createElement("strong", null, arm.average_reward.toFixed(2));
    value.appendChild(strong);
    value.appendChild(
      createElement("span", null, "  ·  " + arm.pulls + (arm.pulls === 1 ? " pull" : " pulls"))
    );
    row.appendChild(value);

    row.title =
      name + " — average reward " + arm.average_reward.toFixed(3) + " over " + arm.pulls + " pulls";
    container.appendChild(row);

    requestAnimationFrame(function () {
      var share = Math.abs(arm.average_reward) / largestMagnitude;
      fill.style.width = Math.max(2, Math.round(share * 100)) + "%";
    });

    // The same numbers, without relying on colour or length.
    var tableRow = createElement("tr");
    tableRow.appendChild(createElement("td", null, name));
    tableRow.appendChild(createElement("td", null, arm.pulls));
    tableRow.appendChild(createElement("td", null, arm.average_reward.toFixed(3)));
    tableBody.appendChild(tableRow);
  });
}

/* ---------------------------------------------------------------------------
   6. Running one ticket
   --------------------------------------------------------------------------- */

function setBusy(busy) {
  isBusy = busy;
  var button = byId("run-button");
  button.disabled = busy;
  button.classList.toggle("is-running", busy);
  button.querySelector(".run-label").textContent = busy ? "Running…" : "Run pipeline";
}

function showFormError(message) {
  var element = byId("form-error");
  if (!message) {
    element.hidden = true;
    return;
  }
  element.textContent = message;
  element.hidden = false;
}

async function runTicket(event) {
  if (event) {
    event.preventDefault();
  }
  if (isBusy) {
    return;
  }

  var subject = byId("subject").value.trim();
  var body = byId("body").value.trim();
  var tier = document.querySelector(".tier.is-selected").dataset.tier;

  if (subject === "" || body === "") {
    showFormError("A subject and a body are both required.");
    return;
  }

  showFormError(null);
  setBusy(true);
  resetDag();

  try {
    var result = await apiPost("/ticket", {
      subject: subject,
      body: body,
      customer_tier: tier,
    });

    // The per-stage timings live in the status endpoint, which reads the
    // workflow engine's own state store.
    var status = await apiGet("/ticket/" + result.transaction_id + "/status", true);

    var note = byId("replay-note");
    note.textContent =
      "Replaying real stage timings (" +
      formatSeconds(result.latency_seconds) +
      " end to end), slowed " +
      REPLAY_SLOWDOWN +
      "× so it is visible.";
    note.hidden = false;

    await replayDag(status.stages);

    renderResult(result);
    await refreshRlDashboard(result.rl_state_key);

    byId("result-panel").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showFormError(error.message);
    markStage("run_agent", "failed");
  } finally {
    setBusy(false);
  }
}

async function sendFeedback(score) {
  if (!currentTransactionId) {
    return;
  }

  byId("fb-up").disabled = true;
  byId("fb-down").disabled = true;

  try {
    var response = await apiPost("/feedback", {
      transaction_id: currentTransactionId,
      feedback_score: score,
    });

    var element = byId("feedback-result");
    clear(element);
    element.appendChild(
      document.createTextNode(
        "reward = " + score + " × 10 − " + response.latency_seconds.toFixed(3) + " = "
      )
    );
    var rewardText = createElement("strong", null, response.reward.toFixed(3));
    rewardText.style.color = response.reward >= 0 ? "var(--status-good)" : "var(--status-critical)";
    element.appendChild(rewardText);
    element.appendChild(
      document.createTextNode(
        " · best for " + response.state_key + " is now " + response.best_config_for_state
      )
    );
    element.hidden = false;

    await refreshRlDashboard(response.state_key);
  } catch (error) {
    var errorElement = byId("feedback-result");
    errorElement.textContent = error.message;
    errorElement.hidden = false;
  }
}

/* ---------------------------------------------------------------------------
   7. The guided "watch it learn" demo
   --------------------------------------------------------------------------- */

var DEMO_TICKET_COUNT = 30;

async function runLearningDemo() {
  if (isBusy) {
    return;
  }

  var button = byId("demo-button");
  var preferred = byId("demo-preference").value;

  setBusy(true);
  button.disabled = true;
  byId("demo-progress").hidden = false;
  clear(byId("demo-picks"));

  // Every ticket in the demo is identical, so they all land in the same RL
  // state and the bandit's choices are directly comparable.
  var ticket = {
    subject: "Charged twice on order 4471",
    body: "My card was charged twice and I want a refund. Nobody has replied for days.",
    customer_tier: "enterprise",
  };

  var liked = 0;
  var stateKey = null;

  try {
    for (var index = 0; index < DEMO_TICKET_COUNT; index = index + 1) {
      var result = await apiPost("/ticket", ticket, true);
      var configName = result.pipeline_config.name;
      stateKey = result.rl_state_key;

      // The simulated customer only likes one prompt variant.
      var score = configName.indexOf(preferred) === 0 ? 1 : 0;
      if (score === 1) {
        liked = liked + 1;
      }

      await apiPost(
        "/feedback",
        { transaction_id: result.transaction_id, feedback_score: score },
        true
      );

      // One small square per ticket, coloured by the arm that was chosen.
      var pick = createElement("span", "demo-pick");
      pick.style.background = CONFIG_COLORS[configName];
      pick.title = "ticket " + (index + 1) + ": " + configName + " (feedback " + score + ")";
      byId("demo-picks").appendChild(pick);

      var done = index + 1;
      byId("demo-bar-fill").style.width = (done / DEMO_TICKET_COUNT) * 100 + "%";
      byId("demo-status").textContent =
        done +
        " / " +
        DEMO_TICKET_COUNT +
        " tickets · " +
        liked +
        " landed on a " +
        preferred +
        " configuration";

      await refreshRlDashboard(stateKey);
    }

    logRequest("DEMO", "30 × /ticket + /feedback", true, "state " + stateKey);

    var lastTen = Array.prototype.slice.call(byId("demo-picks").children, -10);
    var lastTenLiked = lastTen.filter(function (node) {
      return node.title.indexOf("feedback 1") !== -1;
    }).length;

    byId("demo-status").textContent =
      "Done. " +
      liked +
      " of " +
      DEMO_TICKET_COUNT +
      " overall landed on a " +
      preferred +
      " configuration — and " +
      lastTenLiked +
      " of the last 10 did, which is the bandit having learned.";
  } catch (error) {
    byId("demo-status").textContent = "Demo stopped: " + error.message;
  } finally {
    button.disabled = false;
    setBusy(false);
  }
}

/* ---------------------------------------------------------------------------
   8. Start-up
   --------------------------------------------------------------------------- */

function buildSampleButtons() {
  var container = byId("samples");

  SAMPLE_TICKETS.forEach(function (sample) {
    var button = createElement("button", "sample");
    button.type = "button";
    button.style.setProperty("--sample-hue", sample.hue);
    button.appendChild(createElement("span", null, sample.emoji));
    button.appendChild(createElement("span", null, sample.label));

    button.addEventListener("click", function () {
      byId("subject").value = sample.subject;
      byId("body").value = sample.body;

      var tiers = document.querySelectorAll(".tier");
      for (var i = 0; i < tiers.length; i = i + 1) {
        tiers[i].classList.toggle("is-selected", tiers[i].dataset.tier === sample.tier);
      }
      showFormError(null);
    });

    container.appendChild(button);
  });
}

function wireTierButtons() {
  var tiers = document.querySelectorAll(".tier");
  for (var i = 0; i < tiers.length; i = i + 1) {
    tiers[i].addEventListener("click", function (event) {
      var all = document.querySelectorAll(".tier");
      for (var j = 0; j < all.length; j = j + 1) {
        all[j].classList.remove("is-selected");
      }
      event.currentTarget.classList.add("is-selected");
    });
  }
}

async function loadHealth() {
  var statusPill = byId("pill-status");

  try {
    var health = await apiGet("/health", true);

    statusPill.className = "pill";
    clear(statusPill);
    statusPill.appendChild(createElement("span", "dot"));
    statusPill.appendChild(createElement("span", null, health.ready ? "service ready" : "starting"));

    function fill(id, value) {
      var pill = byId(id);
      pill.hidden = false;
      pill.querySelector(".pill-value").textContent = value;
    }

    fill("pill-backend", health.llm_backend);
    fill("pill-chunks", health.knowledge_chunks);
    fill(
      "pill-accuracy",
      health.classifier_accuracy === null
        ? "—"
        : Math.round(health.classifier_accuracy * 100) + "% accurate"
    );
    fill("pill-updates", health.bandit_updates);
  } catch (error) {
    statusPill.className = "pill pill-error";
    clear(statusPill);
    statusPill.appendChild(createElement("span", "dot"));
    statusPill.appendChild(createElement("span", null, "API unreachable"));
  }
}

async function start() {
  buildSampleButtons();
  wireTierButtons();

  byId("ticket-form").addEventListener("submit", runTicket);
  byId("fb-up").addEventListener("click", function () {
    sendFeedback(1);
  });
  byId("fb-down").addEventListener("click", function () {
    sendFeedback(0);
  });
  byId("demo-button").addEventListener("click", runLearningDemo);
  byId("state-select").addEventListener("change", function (event) {
    renderBars(event.target.value);
  });

  await loadHealth();

  try {
    await buildDag();
  } catch (error) {
    byId("dag").appendChild(
      createElement("p", "bars-empty", "Could not load the workflow graph: " + error.message)
    );
  }

  await refreshRlDashboard();

  // Pre-fill the first sample so the page is one click from a demo.
  byId("subject").value = SAMPLE_TICKETS[0].subject;
  byId("body").value = SAMPLE_TICKETS[0].body;
  var tiers = document.querySelectorAll(".tier");
  for (var i = 0; i < tiers.length; i = i + 1) {
    tiers[i].classList.toggle("is-selected", tiers[i].dataset.tier === SAMPLE_TICKETS[0].tier);
  }
}

document.addEventListener("DOMContentLoaded", start);
