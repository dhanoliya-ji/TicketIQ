/* ===========================================================================
   The Performance page.

   Three questions, one card each:
     how accurately does it classify   -> GET /ml/report
     what has it learned to route to   -> GET /rl/stats
     what does a ticket run through    -> GET /workflow/graph

   No numbers are computed here. Everything on the page is served by the API
   and rendered as it arrives, so the page cannot drift from the service.
   =========================================================================== */

"use strict";

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

function percent(value) {
  return Math.round(Number(value) * 1000) / 10 + "%";
}

async function apiGet(path) {
  var response = await fetch(path);
  if (!response.ok) {
    throw new Error("could not load " + path + " (HTTP " + response.status + ")");
  }
  return response.json();
}

function showEmpty(container, message, detail) {
  clear(container);
  var block = make("p", "empty");
  block.appendChild(make("strong", null, message));
  if (detail) {
    block.appendChild(document.createTextNode(" " + detail));
  }
  container.appendChild(block);
}

/* ---------------------------------------------------------------------------
   Statistic tiles
   --------------------------------------------------------------------------- */
function renderStats(container, entries) {
  clear(container);
  entries.forEach(function (entry) {
    var tile = make("article", "stat");
    tile.appendChild(make("p", "stat-label", entry.label));
    tile.appendChild(make("p", "stat-value", entry.value));
    if (entry.note) {
      tile.appendChild(make("p", "stat-note", entry.note));
    }
    container.appendChild(tile);
  });
}

/* ---------------------------------------------------------------------------
   1. Classification quality
   --------------------------------------------------------------------------- */
async function loadClassifier() {
  var report = await apiGet("/ml/report");
  var evaluation = report.evaluation;

  renderStats(byId("classifier-stats"), [
    {
      label: "Accuracy",
      value: percent(evaluation.accuracy),
      note: evaluation.test_size + " held-out tickets",
    },
    { label: "Precision", value: percent(evaluation.macro_precision), note: "macro average" },
    { label: "Recall", value: percent(evaluation.macro_recall), note: "macro average" },
    { label: "F1", value: percent(evaluation.macro_f1), note: "macro average" },
  ]);

  // --- per category ---
  var tableBody = byId("per-category").querySelector("tbody");
  clear(tableBody);

  var categories = Object.keys(evaluation.per_category).sort();
  categories.forEach(function (category) {
    var metrics = evaluation.per_category[category];
    var row = make("tr");

    row.appendChild(make("td", "name", category));
    [metrics.precision, metrics.recall, metrics.f1].forEach(function (value) {
      row.appendChild(make("td", "numeric", value.toFixed(3)));
    });
    row.appendChild(make("td", "numeric cell-muted", metrics.support));
    tableBody.appendChild(row);
  });

  renderConfusion(report.confusion_matrix, categories);
}

function renderConfusion(matrix, categories) {
  var table = byId("confusion");
  clear(table);

  var head = make("tr");
  var corner = make("th", null, "");
  corner.scope = "col";
  head.appendChild(corner);

  categories.forEach(function (category) {
    var cell = make("th", null, titleCase(category));
    cell.scope = "col";
    head.appendChild(cell);
  });

  var thead = make("thead");
  thead.appendChild(head);
  table.appendChild(thead);

  var body = make("tbody");
  categories.forEach(function (actual) {
    var row = make("tr");
    var label = make("th", null, actual);
    label.scope = "row";
    row.appendChild(label);

    categories.forEach(function (predicted) {
      var count = matrix[actual][predicted];
      var cell = make("td", null, count);
      if (actual === predicted) {
        cell.dataset.diagonal = "true";
      } else if (count > 0) {
        // A mistake worth the eye landing on.
        cell.dataset.miss = "true";
      }
      cell.title = count + " " + actual + " ticket(s) predicted as " + predicted;
      row.appendChild(cell);
    });
    body.appendChild(row);
  });
  table.appendChild(body);
}

/* ---------------------------------------------------------------------------
   2. Routing
   --------------------------------------------------------------------------- */
var routingStats = null;

async function loadRouting() {
  routingStats = await apiGet("/rl/stats");

  var stateNames = Object.keys(routingStats.states).sort();
  var decisions = routingStats.exploration_count + routingStats.exploitation_count;

  renderStats(byId("routing-stats"), [
    {
      label: "Ratings received",
      value: routingStats.total_updates,
      note: "each one moves the bandit",
    },
    {
      label: "Ticket types seen",
      value: stateNames.length,
      note: "of 36 possible",
    },
    {
      label: "Explored",
      value: routingStats.exploration_count,
      note: decisions ? percent(routingStats.exploration_count / decisions) + " of choices" : "—",
    },
    {
      label: "Exploited",
      value: routingStats.exploitation_count,
      note: "used the best known setup",
    },
  ]);

  var select = byId("state-select");
  clear(select);

  if (stateNames.length === 0) {
    select.disabled = true;
    byId("routing-scope").textContent = "Nothing learned yet.";
    showEmpty(
      byId("routing-body"),
      "No tickets have been routed yet.",
      "Triage a ticket and rate the reply, and this fills in."
    );
    return;
  }

  select.disabled = false;
  stateNames.forEach(function (name) {
    var option = make("option", null, describeState(name));
    option.value = name;
    select.appendChild(option);
  });

  // Open on the ticket type with the most evidence behind it.
  select.value = busiestState(stateNames);
  renderRoutingTable(select.value);
}

/* "billing|high|enterprise" reads as a key, not as a sentence. */
function describeState(stateKey) {
  var parts = stateKey.split("|");
  return (
    titleCase(parts[0]) + " · " + parts[1] + " urgency · " + parts[2] + " tier"
  );
}

function busiestState(stateNames) {
  var best = stateNames[0];
  var bestPulls = -1;

  stateNames.forEach(function (name) {
    var arms = routingStats.states[name];
    var pulls = 0;
    Object.keys(arms).forEach(function (action) {
      pulls = pulls + arms[action].pulls;
    });
    if (pulls > bestPulls) {
      bestPulls = pulls;
      best = name;
    }
  });
  return best;
}

function renderRoutingTable(stateKey) {
  var arms = routingStats.states[stateKey];
  var models = {};
  routingStats.configurations.forEach(function (config) {
    models[config.name] = config.model;
  });

  var rated = 0;
  var bestName = null;
  var bestReward = -Infinity;
  var largest = 0.001;

  routingStats.actions.forEach(function (name) {
    var arm = arms[name];
    rated = rated + arm.pulls;
    largest = Math.max(largest, Math.abs(arm.average_reward));
    // Only an arm that has actually been tried can be the best one.
    if (arm.pulls > 0 && arm.average_reward > bestReward) {
      bestReward = arm.average_reward;
      bestName = name;
    }
  });

  byId("routing-scope").textContent =
    describeState(stateKey) + " — " + rated + (rated === 1 ? " rating" : " ratings");

  // Negative rewards are normal (reward subtracts latency), so the bar needs a
  // zero line in the middle whenever anything is below zero.
  var anyNegative = routingStats.actions.some(function (name) {
    return arms[name].average_reward < 0;
  });
  var zeroAt = anyNegative ? 50 : 0;

  var tableBody = byId("routing-table").querySelector("tbody");
  clear(tableBody);

  routingStats.actions.forEach(function (name) {
    var arm = arms[name];
    var row = make("tr");
    if (name === bestName) {
      row.dataset.best = "true";
    }

    var nameCell = make("td", "name");
    nameCell.appendChild(document.createTextNode(name));
    if (name === bestName) {
      nameCell.appendChild(document.createTextNode(" "));
      nameCell.appendChild(make("span", "pill", "best"));
    }
    row.appendChild(nameCell);

    row.appendChild(make("td", "cell-muted", models[name] || "—"));

    // --- the bar ---
    var barCell = make("td", "bar-cell");
    var track = make("div", "bar-track");
    var zero = make("div", "bar-zero");
    zero.style.left = zeroAt + "%";
    track.appendChild(zero);

    if (arm.pulls > 0) {
      var fill = make("div", "bar-fill");
      var share = (Math.abs(arm.average_reward) / largest) * (anyNegative ? 50 : 100);
      if (arm.average_reward < 0) {
        fill.dataset.sign = "negative";
        fill.style.right = 100 - zeroAt + "%";
        fill.style.width = share + "%";
      } else {
        fill.style.left = zeroAt + "%";
        fill.style.width = share + "%";
      }
      track.appendChild(fill);
    }
    barCell.appendChild(track);
    row.appendChild(barCell);

    // --- the number, which is what makes the bar readable without colour ---
    var value = make(
      "td",
      "numeric " + (arm.pulls === 0 ? "cell-muted" : arm.average_reward < 0 ? "cell-critical" : ""),
      arm.pulls === 0 ? "—" : arm.average_reward.toFixed(2)
    );
    row.appendChild(value);

    row.appendChild(make("td", "numeric cell-muted", arm.pulls));
    tableBody.appendChild(row);
  });
}

/* ---------------------------------------------------------------------------
   3. Pipeline
   --------------------------------------------------------------------------- */
async function loadPipeline() {
  var graph = await apiGet("/workflow/graph");
  var container = byId("pipeline");
  clear(container);

  graph.levels.forEach(function (level, index) {
    if (index > 0) {
      container.appendChild(make("div", "pipeline-arrow", "↓"));
    }

    var row = make("div", "pipeline-level");
    row.dataset.parallel = level.runs_in_parallel ? "true" : "false";

    level.stages.forEach(function (stage) {
      var block = make("div", "pipeline-stage");
      block.appendChild(make("span", "pipeline-name", stage.name));
      block.appendChild(make("span", "pipeline-desc", stage.description));
      row.appendChild(block);
    });

    if (level.runs_in_parallel) {
      row.appendChild(make("span", "pipeline-tag", "run at the same time"));
    }
    container.appendChild(row);
  });
}

/* ---------------------------------------------------------------------------
   Start-up
   --------------------------------------------------------------------------- */
async function start() {
  byId("state-select").addEventListener("change", function (event) {
    renderRoutingTable(event.target.value);
  });

  // Each card fails on its own, so one unavailable endpoint does not blank
  // the whole page.
  var sections = [
    ["classifier-stats", loadClassifier],
    ["routing-body", loadRouting],
    ["pipeline", loadPipeline],
  ];

  for (var index = 0; index < sections.length; index = index + 1) {
    var containerId = sections[index][0];
    try {
      await sections[index][1]();
    } catch (error) {
      showEmpty(byId(containerId), "Could not load this section.", error.message);
    }
  }
}

document.addEventListener("DOMContentLoaded", start);
