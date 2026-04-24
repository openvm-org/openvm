// Recursion Sub-Circuit Interactive Guide — frontend logic

(function () {
  const D = window.DATA;

  // Polyfill CSS.escape for older/alt runtimes (jsdom etc.). Modern browsers have it natively.
  const cssEscape =
    (typeof CSS !== "undefined" && CSS.escape)
      ? (s) => CSS.escape(s)
      : (s) => String(s).replace(/[^a-zA-Z0-9_-]/g, (c) => "\\" + c);

  // --- Lookups ---
  const airsById = {};
  const airsByName = {}; // name -> array of AIRs (for disambiguation)
  const busesByName = {};
  D.airs.forEach((a) => {
    airsById[a.id] = a;
    (airsByName[a.name] = airsByName[a.name] || []).push(a);
  });
  D.buses.forEach((b) => (busesByName[b.name] = b));

  // Build producers/consumers index per bus for quick lookups
  D.buses.forEach((b) => {
    b._airSet = new Set();
    (b.producers || []).forEach((p) => b._airSet.add(p.air));
    (b.consumers || []).forEach((c) => b._airSet.add(c.air));
  });

  // --- Module metadata ---
  const MODULE_LABEL = {
    transcript: "Transcript",
    "proof-shape": "ProofShape",
    primitives: "Primitives",
    gkr: "GKR",
    "batch-constraint": "BatchConstraint",
    stacking: "Stacking",
    whir: "WHIR",
  };
  const MODULE_ORDER = [
    "transcript",
    "proof-shape",
    "primitives",
    "gkr",
    "batch-constraint",
    "stacking",
    "whir",
  ];
  const PHASE_LABEL = {
    preamble: "Phase 1 — Preamble",
    gkr: "Phase 2a — GKR",
    bc: "Phase 2b — Batch Constraint",
    stacking: "Phase 3 — Stacking",
    whir: "Phase 4 — WHIR",
  };

  // --- Utilities ---
  const $ = (id) => document.getElementById(id);
  const el = (tag, attrs, children) => {
    const e = document.createElement(tag);
    if (attrs) {
      for (const k in attrs) {
        if (k === "class") e.className = attrs[k];
        else if (k === "html") e.innerHTML = attrs[k];
        else if (k === "text") e.textContent = attrs[k];
        else if (k.startsWith("data-")) e.setAttribute(k, attrs[k]);
        else if (k === "onclick") e.addEventListener("click", attrs[k]);
        else if (k === "onmouseenter") e.addEventListener("mouseenter", attrs[k]);
        else if (k === "onmouseleave") e.addEventListener("mouseleave", attrs[k]);
        else e[k] = attrs[k];
      }
    }
    if (children) {
      (Array.isArray(children) ? children : [children]).forEach((c) => {
        if (c == null) return;
        e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
      });
    }
    return e;
  };
  const esc = (s) =>
    String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // --- Auto-linking: wrap AIR/bus/column names in refs ---
  // Build a single regex alternation of all known names; longer first so
  // MultilinearSumcheckAir matches before SumcheckAir.
  const NAME_TO_REF = [];
  Object.keys(airsByName).forEach((name) => {
    const airs = airsByName[name];
    // If multiple AIRs share the name (e.g. SumcheckAir), we point to the first; callers should use canonical ids when precision matters.
    NAME_TO_REF.push({ name, type: "air", id: airs[0].id });
  });
  Object.keys(busesByName).forEach((name) => {
    NAME_TO_REF.push({ name, type: "bus", id: name });
  });
  NAME_TO_REF.sort((a, b) => b.name.length - a.name.length);
  const NAME_REGEX = new RegExp(
    "\\b(" + NAME_TO_REF.map((r) => r.name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|") + ")\\b",
    "g"
  );
  const NAME_MAP = {};
  NAME_TO_REF.forEach((r) => (NAME_MAP[r.name] = r));

  function autoLink(text, columnNames) {
    if (!text) return "";
    // First escape, then replace matched tokens with ref spans.
    let out = esc(text);
    out = out.replace(NAME_REGEX, (m) => {
      const r = NAME_MAP[m];
      if (!r) return m;
      return `<span class="ref" data-type="${r.type}" data-id="${esc(r.id)}">${m}</span>`;
    });
    // Optional column-name linking (only when a columns array is supplied).
    if (columnNames && columnNames.length) {
      const colRe = new RegExp(
        "\\b(" + columnNames.map((n) => n.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|") + ")\\b",
        "g"
      );
      out = out.replace(colRe, (m) => `<span class="ref" data-type="col" data-id="${esc(m)}">${m}</span>`);
    }
    return out;
  }

  // --- Tooltip ---
  const tipEl = $("tooltip");
  let tipTimer = null;
  function showTooltip(html, x, y) {
    tipEl.innerHTML = html;
    tipEl.style.left = x + "px";
    tipEl.style.top = y + "px";
    tipEl.classList.add("visible");
  }
  function hideTooltip() {
    tipEl.classList.remove("visible");
  }
  function tipForAir(air) {
    return `<div class="tip-title">${esc(air.name)} <span class="badge m-${air.module}">${MODULE_LABEL[air.module] || air.module}</span></div>
      <div class="tip-sub">${esc(air.role || "")}</div>`;
  }
  function tipForBus(bus) {
    const fmt = (bus.message_format || []).map((f) => `<tr><td>${esc(f.field)}</td><td>${esc(f.type || "?")}</td></tr>`).join("");
    return `<div class="tip-title">${esc(bus.name)} <span class="kind-pill ${bus.kind}">${esc(bus.kind)}</span></div>
      <div class="tip-sub">${esc(bus.purpose || "")}</div>
      ${fmt ? `<table>${fmt}</table>` : ""}`;
  }
  function tipForColumn(name) {
    return `<div class="tip-title mono">${esc(name)}</div><div class="tip-sub">Trace column</div>`;
  }

  // --- Section switching ---
  function showSection(id) {
    document.querySelectorAll(".section").forEach((s) => s.classList.remove("active"));
    $("section-" + id).classList.add("active");
    document.querySelectorAll("#main-nav button").forEach((b) => {
      b.classList.toggle("active", b.dataset.section === id);
    });
    document.body.classList.toggle("section-proof", id === "proof");
    window.scrollTo({ top: 0 });
    // Viz panel connections depend on real element coords; when the proof
    // section was hidden its tables had zero size. Re-apply the current scene
    // after layout settles so arrows land on the right cells.
    if (id === "proof" && currentStepId) {
      setTimeout(() => {
        renderStepViz({ id: currentStepId, airs: [], buses: [] });
      }, 30);
    }
  }
  document.querySelectorAll("#main-nav button").forEach((b) => {
    b.addEventListener("click", () => showSection(b.dataset.section));
  });

  // --- Overview ---
  function renderOverview() {
    $("stat-airs").textContent = D.airs.length;
    $("stat-buses").textContent = D.buses.length;
    const stepCount = D.phases.reduce((s, p) => s + p.steps.length, 0);
    $("stat-steps").textContent = stepCount;

    // Claim + decomposition
    const mc = D.main_claim;
    $("claim-box").innerHTML = `
      <div class="ov-card" style="padding:18px 22px">
        <p style="margin-top:0">${autoLink(mc.statement)}</p>
        <div class="section-label">Assumptions the enclosing circuit must verify</div>
        <ol style="margin: 6px 0 0 20px; padding: 0">
          ${mc.assumptions.map((a) => `<li>${autoLink(a)}</li>`).join("")}
        </ol>
      </div>
    `;

    const decomp = mc.decomposition || [];
    $("decomp-box").innerHTML = `
      <p class="muted">The main claim splits into these sub-claims. Each is discharged by a dedicated module or lookup table (follow links in the Proof Walkthrough).</p>
      <div class="overview-grid" style="grid-template-columns: repeat(auto-fit, minmax(340px, 1fr))">
        ${decomp
          .map(
            (d) =>
              `<div class="ov-card"><h4 class="mono" style="font-size:0.92rem">${esc(d.label)}</h4><div class="small muted" style="margin-top:6px">${autoLink(
                d.statement
              )}</div></div>`
          )
          .join("")}
      </div>
    `;

    // Arch node click → jump to AIR browser filtered by module
    document.querySelectorAll(".arch-node[data-module]").forEach((n) => {
      n.addEventListener("click", () => {
        filterAirsByModule(n.dataset.module);
        showSection("airs");
      });
    });
  }

  // --- Proof walkthrough (chapter mode) ---
  //
  // Each phase is a single scrollable page composed of blocks:
  //   { type: "prose", html }
  //   { type: "air_header", heading, html, air_id?, source_path? }
  //   { type: "code", heading, source_path, source_lines, code, explain, scene?, lang? }
  //
  // A code block's optional `scene` is a key into D.scenes (same keys used by
  // the old step-bundle system: preamble-1, gkr-5, bc-11, etc.). Hovering a
  // code block activates that scene in the viz panel; clicking pins it.
  //
  // The viz panel is re-rendered lazily when the active scene changes, because
  // different scenes reference different AIRs/buses.

  let currentChapterId = null;
  // `currentStepId` (legacy name) now holds the current SCENE KEY so that
  // renderStepViz() / applyScene() / the resize handler still work unchanged.
  let currentStepId = null;

  const CHAPTER_ORDER = ["preamble", "gkr", "bc", "stacking", "whir"];

  function getChapter(phaseId) {
    return (D.chapters && D.chapters[phaseId]) || null;
  }

  function renderPhaseTabs() {
    const container = $("phase-tabs");
    if (!container) return;
    container.innerHTML = "";
    CHAPTER_ORDER.forEach((pid) => {
      const ch = getChapter(pid);
      const label = PHASE_LABEL[pid] || pid;
      const [main, sub] = label.split(/\s*—\s*/);
      const btn = el("button", {
        class: "phase-tab",
        "data-phase-id": pid,
        role: "tab",
        title: ch ? ch.name : label,
      }, [
        el("span", { text: sub || main }),
        el("span", { class: "phase-tab-sub", text: sub ? main : "" }),
      ]);
      btn.addEventListener("click", () => selectChapter(pid));
      container.appendChild(btn);
    });
  }

  function selectChapter(phaseId) {
    currentChapterId = phaseId;
    // Reset pin/hover state on chapter switch.
    if (pinnedRefEl) pinnedRefEl.classList.remove("pinned");
    pinnedSceneKey = null;
    pinnedRefEl = null;
    currentSceneKey = null;

    document.querySelectorAll("#phase-tabs .phase-tab").forEach((b) => {
      b.classList.toggle("active", b.dataset.phaseId === phaseId);
    });
    renderChapter(phaseId);

    // Latch the viz panel onto the first code block with a scene (so the panel
    // isn't empty on page load).
    const firstScene = document.querySelector(
      `.chapter-content .code-block[data-scene]`
    );
    if (firstScene) {
      activateSceneBlock(firstScene, false);
    } else {
      renderVizEmpty();
    }
    window.scrollTo({ top: 0 });
  }

  function renderVizEmpty() {
    currentStepId = null;
    const viz = $("proof-viz");
    if (!viz) return;
    viz.innerHTML = "";
    viz.appendChild(
      el("div", {
        class: "viz-empty",
        text:
          "Hover over a code chunk on the left to see the cells it constrains and the bus connections it drives. Click a chunk to pin the view.",
      })
    );
  }

  // Minimal Rust token highlighter — cosmetic only.
  const RUST_KEYWORDS = new Set([
    "as","async","await","break","const","continue","crate","dyn","else","enum","extern","false","fn","for","if","impl","in","let","loop","match","mod","move","mut","pub","ref","return","self","Self","static","struct","super","trait","true","type","unsafe","use","where","while","box","yield","abstract","become","do","final","macro","override","priv","typeof","unsized","virtual","try",
  ]);
  const RUST_TYPES = new Set([
    "bool","char","f32","f64","i8","i16","i32","i64","i128","isize","str","u8","u16","u32","u64","u128","usize","String","Vec","Option","Result","Box","Rc","Arc","Self",
  ]);
  function highlightRust(code) {
    // Work on a pre-escaped string; tokens are added via span classes.
    const esc1 = esc(code);
    // Comments (line and block)
    let out = esc1
      .replace(/(\/\/[^\n]*)/g, '<span class="c">$1</span>')
      .replace(/(\/\*[\s\S]*?\*\/)/g, '<span class="c">$1</span>');
    // Strings (simple; double-quoted)
    out = out.replace(/(&quot;[^&]*?&quot;)/g, '<span class="s">$1</span>');
    // Attributes / macros: #[...] or foo!
    out = out.replace(/(#!?\[[^\]]*\])/g, '<span class="a">$1</span>');
    out = out.replace(/\b([a-zA-Z_][a-zA-Z0-9_]*!)/g, '<span class="a">$1</span>');
    // Numbers
    out = out.replace(/\b(\d+(?:\.\d+)?(?:_\w+)?)\b/g, '<span class="n">$1</span>');
    // Keywords / types — avoid re-wrapping already-wrapped spans.
    out = out.replace(/\b([A-Za-z_][A-Za-z0-9_]*)\b/g, (m) => {
      if (RUST_KEYWORDS.has(m)) return `<span class="k">${m}</span>`;
      if (RUST_TYPES.has(m)) return `<span class="t">${m}</span>`;
      return m;
    });
    return out;
  }

  function renderChapter(phaseId) {
    const root = $("chapter-content");
    if (!root) return;
    root.innerHTML = "";
    const ch = getChapter(phaseId);
    if (!ch) {
      root.appendChild(
        el("div", { class: "viz-empty", text: `No chapter data loaded for phase "${phaseId}".` })
      );
      return;
    }
    root.appendChild(el("h2", { class: "chapter-title", text: ch.name || phaseId }));
    if (ch.summary) {
      root.appendChild(el("div", { class: "chapter-summary", html: autoLink(ch.summary) }));
    }

    (ch.blocks || []).forEach((blk, i) => {
      if (blk.type === "prose") {
        root.appendChild(
          el("div", { class: "prose-block", html: blk.html || "" })
        );
      } else if (blk.type === "air_header") {
        const hdr = el("div", { class: "air-header" });
        hdr.appendChild(el("h3", { text: blk.heading || blk.air_id || "" }));
        if (blk.source_path) {
          hdr.appendChild(
            el("div", { class: "air-src mono", text: blk.source_path })
          );
        }
        if (blk.html) {
          hdr.appendChild(el("div", { class: "air-desc", html: blk.html }));
        }
        root.appendChild(hdr);
      } else if (blk.type === "code") {
        root.appendChild(renderCodeBlock(blk, phaseId, i));
      }
    });
  }

  function renderCodeBlock(blk, phaseId, idx) {
    const hasScene = !!blk.scene;
    const wrap = el("div", {
      class: "code-block" + (hasScene ? "" : " no-scene"),
      "data-phase-id": phaseId,
      "data-block-idx": String(idx),
    });
    if (hasScene) wrap.setAttribute("data-scene", blk.scene);

    // Header
    const header = el("div", { class: "code-block-header" });
    const left = el("div");
    if (blk.heading) left.appendChild(el("span", { class: "cb-heading", text: blk.heading }));
    const right = el("div", {}, [
      blk.source_lines
        ? el("span", {
            class: "cb-src",
            text:
              (blk.source_path ? blk.source_path.split("/").pop() : "") +
              ":" +
              blk.source_lines,
          })
        : null,
      hasScene ? el("span", { class: "cb-pinned-pill", text: "pinned" }) : null,
    ]);
    header.appendChild(left);
    header.appendChild(right);
    wrap.appendChild(header);

    // Code
    const pre = el("pre");
    const codeEl = el("code", { class: "language-" + (blk.lang || "rust") });
    codeEl.innerHTML = blk.lang === "rust" || !blk.lang
      ? highlightRust(blk.code || "")
      : esc(blk.code || "");
    pre.appendChild(codeEl);
    wrap.appendChild(pre);

    // Explain
    if (blk.explain) {
      wrap.appendChild(
        el("div", { class: "cb-explain", html: blk.explain })
      );
    }
    return wrap;
  }

  // Activate a code block's scene in the viz panel.
  // When `fromClick` is false we only update hover state; pin state is preserved.
  function activateSceneBlock(blockEl, fromClick) {
    const sceneKey = blockEl && blockEl.dataset && blockEl.dataset.scene;
    if (!sceneKey) return;
    // Don't override a pinned scene on hover.
    if (!fromClick && pinnedSceneKey) return;
    currentSceneKey = null; // default scene of the bundle
    currentStepId = sceneKey;
    document.querySelectorAll(".code-block.active").forEach((e) => {
      if (e !== blockEl) e.classList.remove("active");
    });
    blockEl.classList.add("active");
    // renderStepViz uses D.scenes[step.id] to pick AIRs/buses; pass a synthetic step.
    renderStepViz({ id: sceneKey, airs: [], buses: [] });
  }

  // ---------------------------------------------------------------------------
  // Scene-aware proof viz panel (v2).
  //
  // Each step may have a "scene bundle" in D.scenes[stepId] of the form:
  //   { default_scene, scenes: { key: Scene, ... }, constraints_html?, soundness_html? }
  // A Scene is { airs, buses, connections, note }.
  // When the user hovers a <span class="ref" data-scene="X"> the panel re-applies
  // that scene; on leave, we return to the default.
  // ---------------------------------------------------------------------------

  // Short label for a column name — keep table cells compact.
  function shortColLabel(name) {
    if (!name) return "";
    // single-word short names pass through; longer snake_case names are abbreviated to initials.
    if (name.length <= 10) return name;
    const parts = name.split(/[_\s]+/).filter(Boolean);
    if (parts.length === 1) return name.slice(0, 10);
    return parts.map((p) => p[0]).join("").toLowerCase();
  }

  // Short subscript for a row index — row 0/1/2/-1 → c_0, c_1, c_2, c_{n-1}
  function subscriptFor(row) {
    if (row === -1) return "{n-1}";
    return String(row);
  }

  // Placeholder cell content like c_0, c_{n-1}. Uses simple dot notation.
  function cellPlaceholder(colName, row) {
    const base = shortColLabel(colName);
    const sub = subscriptFor(row);
    return `${base}_${sub}`;
  }

  // Clamp scene row → a rendered row (0, 1, 2, -1). Intermediate rows map to "ellipsis".
  function clampRow(row, renderedRows) {
    if (renderedRows.includes(row)) return row;
    if (row === -1) return -1;
    if (row >= 3) return "ellipsis";
    return renderedRows[0];
  }

  // Render a compact AIR trace table with fixed 4 visible rows: 0, 1, 2, ellipsis, n-1.
  function renderAirTable(air, opts) {
    opts = opts || {};
    const wrap = el("div", { class: "air-viz", "data-air-id": air.id, "data-air-name": air.name });
    const header = el("div", { class: "air-viz-header" }, [
      el("span", { class: "air-name" }, [
        el("span", { class: "badge m-" + air.module, text: MODULE_LABEL[air.module] || air.module }),
        " " + air.name,
      ]),
      el("span", { class: "air-role", text: air.role || "" }),
    ]);
    wrap.appendChild(header);

    const tableWrap = el("div", { class: "air-table-wrap" });
    const table = el("table", { class: "air-trace" });
    const thead = el("thead");
    const tr = el("tr");
    // Row-index label column
    tr.appendChild(el("th", { class: "row-label-col", text: "row" }));
    const cols = air.columns.length ? air.columns : [{ name: "(no columns)", type: "", description: "" }];
    cols.forEach((c) => {
      const th = el("th", {
        "data-col-name": c.name,
        "data-air-col": c.name,
        text: shortColLabel(c.name),
        title: `${c.name}${c.type ? " : " + c.type : ""}${c.description ? " — " + c.description : ""}`,
      });
      tr.appendChild(th);
    });
    thead.appendChild(tr);
    table.appendChild(thead);

    const tbody = el("tbody");
    const visibleRows = [0, 1, 2];
    visibleRows.forEach((r) => {
      const body = el("tr", { "data-air-row": String(r) });
      body.appendChild(el("td", { class: "row-label", text: String(r) }));
      cols.forEach((c) => {
        body.appendChild(
          el("td", {
            "data-air": air.name,
            "data-air-id": air.id,
            "data-col": c.name,
            "data-row": String(r),
            text: cellPlaceholder(c.name, r),
            title: `${air.name}.${c.name}[row ${r}]${c.type ? " : " + c.type : ""}`,
          })
        );
      });
      tbody.appendChild(body);
    });
    // Ellipsis row
    const ell = el("tr", { class: "ellipsis-row" });
    const ellTd = el("td", { text: "⋮" });
    ellTd.setAttribute("colspan", String(cols.length + 1));
    ell.appendChild(ellTd);
    tbody.appendChild(ell);
    // Last row n-1
    const lastRow = el("tr", { "data-air-row": "-1" });
    lastRow.appendChild(el("td", { class: "row-label", text: "n-1" }));
    cols.forEach((c) => {
      lastRow.appendChild(
        el("td", {
          "data-air": air.name,
          "data-air-id": air.id,
          "data-col": c.name,
          "data-row": "-1",
          text: cellPlaceholder(c.name, -1),
          title: `${air.name}.${c.name}[row n-1]${c.type ? " : " + c.type : ""}`,
        })
      );
    });
    tbody.appendChild(lastRow);

    table.appendChild(tbody);
    tableWrap.appendChild(table);
    wrap.appendChild(tableWrap);
    return wrap;
  }

  // Render a compact bus "trace" table (field columns, rows = 2 by default, or whatever scenes reference).
  function renderBusTable(bus, opts) {
    opts = opts || {};
    const rowCount = Math.max(opts.rows || 2, 2);
    const card = el("div", { class: "bus-card", "data-bus-name": bus.name });
    card.appendChild(
      el("div", { class: "bus-card-header" }, [
        el("span", { text: bus.name }),
        el("span", { class: "kind-pill " + bus.kind, text: bus.kind }),
      ])
    );
    const tableWrap = el("div", { class: "bus-table-wrap" });
    const table = el("table", { class: "bus-trace" });
    const thead = el("thead");
    const tr = el("tr");
    tr.appendChild(el("th", { class: "row-label-col", text: "msg" }));
    const fields = (bus.message_format && bus.message_format.length)
      ? bus.message_format
      : [{ field: "(no format)", type: "", description: "" }];
    fields.forEach((f) => {
      tr.appendChild(
        el("th", {
          "data-bus-field": f.field,
          text: f.field,
          title: `${f.field}${f.type ? " : " + f.type : ""}${f.description ? " — " + f.description : ""}`,
        })
      );
    });
    thead.appendChild(tr);
    table.appendChild(thead);

    const tbody = el("tbody");
    for (let r = 0; r < rowCount; r++) {
      const body = el("tr", { "data-bus-row": String(r) });
      body.appendChild(el("td", { class: "row-label", text: String(r) }));
      fields.forEach((f) => {
        body.appendChild(
          el("td", {
            "data-bus": bus.name,
            "data-field": f.field,
            "data-row": String(r),
            text: `${shortColLabel(f.field)}_${r}`,
            title: `${bus.name}.${f.field}[msg ${r}]${f.type ? " : " + f.type : ""}`,
          })
        );
      });
      tbody.appendChild(body);
    }
    table.appendChild(tbody);
    tableWrap.appendChild(table);
    card.appendChild(tableWrap);
    return card;
  }

  // Return the maximum bus-row index (0-based) referenced in a scene bundle for a given bus.
  function maxBusRowInBundle(bundle, busName) {
    let max = 1; // minimum of 2 rows
    const collect = (scene) => {
      if (!scene) return;
      (scene.buses || []).forEach((b) => {
        if (b.name !== busName) return;
        (b.rows || []).forEach((r) => {
          if (typeof r.row === "number" && r.row > max) max = r.row;
        });
      });
      (scene.connections || []).forEach((c) => {
        if (c.to && c.to.bus === busName && typeof c.to.row === "number" && c.to.row > max) max = c.to.row;
      });
    };
    collect(bundle.default_scene);
    Object.values(bundle.scenes || {}).forEach(collect);
    return max + 1;
  }

  function renderStepViz(step) {
    const viz = $("proof-viz");
    viz.innerHTML = "";

    const bundle = (D.scenes && D.scenes[step.id]) || null;

    // Determine set of AIRs/buses to render. Prefer the default_scene, else fall back to step lists.
    const airNames = new Set();
    const busNames = new Set();

    if (bundle) {
      const gather = (scene) => {
        if (!scene) return;
        (scene.airs || []).forEach((a) => airNames.add(a.name));
        (scene.buses || []).forEach((b) => busNames.add(b.name));
      };
      gather(bundle.default_scene);
      Object.values(bundle.scenes || {}).forEach(gather);
    }
    // Also include step-listed AIRs/buses (so scenes don't have to re-enumerate them)
    step.airs.forEach((aid) => {
      const air = airsById[aid];
      if (air) airNames.add(air.name);
    });
    (step.buses || []).forEach((bn) => busNames.add(bn));

    // Note area
    const noteEl = el("div", { class: "viz-note", text: "" });
    viz.appendChild(noteEl);

    // Surface + overlay
    const surface = el("div", { class: "viz-surface" });
    const airsSec = el("div", { class: "viz-section" });
    airsSec.appendChild(el("h3", { text: `AIR traces (${airNames.size})` }));
    Array.from(airNames).forEach((name) => {
      const airs = airsByName[name];
      const air = airs && airs[0];
      if (air) airsSec.appendChild(renderAirTable(air));
      else airsSec.appendChild(el("div", { class: "small muted", text: "Unknown AIR: " + name }));
    });
    surface.appendChild(airsSec);

    if (busNames.size) {
      const busSec = el("div", { class: "viz-section" });
      busSec.appendChild(el("h3", { text: `Buses (${busNames.size})` }));
      const busGrid = el("div", { class: "bus-cards" });
      Array.from(busNames).forEach((bn) => {
        const bus = busesByName[bn];
        if (bus) {
          const rowCount = bundle ? maxBusRowInBundle(bundle, bn) : 2;
          busGrid.appendChild(renderBusTable(bus, { rows: rowCount }));
        } else {
          busGrid.appendChild(el("div", { class: "small muted", text: "Unknown bus: " + bn }));
        }
      });
      busSec.appendChild(busGrid);
      surface.appendChild(busSec);
    }

    // Overlay SVG for connections
    const svgNS = "http://www.w3.org/2000/svg";
    const overlay = document.createElementNS(svgNS, "svg");
    overlay.setAttribute("class", "viz-overlay");
    overlay.setAttribute("preserveAspectRatio", "none");
    surface.appendChild(overlay);

    viz.appendChild(surface);

    // Apply default scene
    applyScene(step.id, null);
  }

  // Clear highlights + SVG connection lines. Also strip any cell-<kind> or field-<kind> class.
  const SCENE_KINDS = ["primary", "prev", "next", "secondary"];
  function clearSceneHighlights() {
    const viz = $("proof-viz");
    if (!viz) return;
    viz.querySelectorAll("td,tr,.air-viz,.bus-card").forEach((e) => {
      SCENE_KINDS.forEach((k) => {
        e.classList.remove("cell-" + k);
        e.classList.remove("field-" + k);
      });
      e.classList.remove("highlight", "ellipsis-highlight");
    });
    const overlay = viz.querySelector("svg.viz-overlay");
    if (overlay) {
      // Preserve <defs> so arrow markers persist across scene switches.
      const defs = overlay.querySelector("defs");
      overlay.innerHTML = "";
      if (defs) overlay.appendChild(defs);
    }
    const note = viz.querySelector(".viz-note");
    if (note) note.textContent = "";
  }

  // Resolve a scene cell (AIR) to a DOM <td>. Clamps rows not rendered (falls back to ellipsis row marker).
  function findAirCell(airName, col, row) {
    const viz = $("proof-viz");
    if (!viz) return null;
    const rowStr = String(row);
    let td = viz.querySelector(
      `td[data-air="${cssEscape(airName)}"][data-col="${cssEscape(col)}"][data-row="${cssEscape(rowStr)}"]`
    );
    if (td) return td;
    // Out-of-range: mark the ellipsis row of that AIR's table (if any) and fall back to the last rendered row.
    const table = viz.querySelector(`.air-viz[data-air-name="${cssEscape(airName)}"] table.air-trace`);
    if (!table) return null;
    const ell = table.querySelector("tr.ellipsis-row");
    if (ell) ell.classList.add("ellipsis-highlight");
    return table.querySelector(`td[data-col="${cssEscape(col)}"]`);
  }

  function findBusCell(busName, field, row) {
    const viz = $("proof-viz");
    if (!viz) return null;
    return viz.querySelector(
      `td[data-bus="${cssEscape(busName)}"][data-field="${cssEscape(field)}"][data-row="${cssEscape(String(row))}"]`
    );
  }

  function applyScene(stepId, sceneKey) {
    const bundle = (D.scenes && D.scenes[stepId]) || null;
    clearSceneHighlights();
    if (!bundle) {
      // Coarse fallback: highlight AIR cards / bus cards referenced in the step.
      const step = findStep(stepId);
      if (!step) return;
      const viz = $("proof-viz");
      step.airs.forEach((aid) => {
        viz.querySelectorAll(`.air-viz[data-air-id="${cssEscape(aid)}"]`).forEach((e) => e.classList.add("highlight"));
      });
      (step.buses || []).forEach((bn) => {
        viz.querySelectorAll(`.bus-card[data-bus-name="${cssEscape(bn)}"]`).forEach((e) => e.classList.add("highlight"));
      });
      return;
    }
    const scene = (sceneKey && bundle.scenes && bundle.scenes[sceneKey]) || bundle.default_scene;
    if (!scene) return;

    // Note
    const note = $("proof-viz").querySelector(".viz-note");
    if (note) note.textContent = scene.note || "";

    // Highlight AIR cells
    (scene.airs || []).forEach((a) => {
      (a.rows || []).forEach((rowSpec) => {
        const kind = rowSpec.kind || "primary";
        (rowSpec.cols || []).forEach((col) => {
          const td = findAirCell(a.name, col, rowSpec.row);
          if (td) td.classList.add("cell-" + kind);
        });
      });
    });

    // Highlight bus cells
    (scene.buses || []).forEach((b) => {
      (b.rows || []).forEach((rowSpec) => {
        const kind = rowSpec.kind || "primary";
        (rowSpec.fields || []).forEach((field) => {
          const td = findBusCell(b.name, field, rowSpec.row);
          if (td) td.classList.add("field-" + kind);
        });
      });
    });

    // Draw connection lines
    drawConnections(scene.connections || []);
  }

  function drawConnections(connections) {
    const viz = $("proof-viz");
    if (!viz) return;
    const overlay = viz.querySelector("svg.viz-overlay");
    const surface = viz.querySelector(".viz-surface");
    if (!overlay || !surface) return;

    const svgNS = "http://www.w3.org/2000/svg";
    const surfaceRect = surface.getBoundingClientRect();
    // Size overlay to cover the full scrollable content area (not just visible),
    // so connection lines reach cells in horizontally-wide tables.
    const surfW = Math.max(surface.scrollWidth, surfaceRect.width);
    const surfH = Math.max(surface.scrollHeight, surfaceRect.height);
    overlay.setAttribute("width", String(surfW));
    overlay.setAttribute("height", String(surfH));
    overlay.setAttribute("viewBox", `0 0 ${surfW} ${surfH}`);
    overlay.style.width = surfW + "px";
    overlay.style.height = surfH + "px";

    // Arrow markers (one per kind) — lazily injected once.
    ensureSvgDefs(overlay, svgNS);

    connections.forEach((c) => {
      if (!c.from || !c.to) return;
      const fromEl = findAirCell(c.from.air, c.from.col, c.from.row);
      const toEl = findBusCell(c.to.bus, c.to.field, c.to.row);
      if (!fromEl || !toEl) return;
      const fr = fromEl.getBoundingClientRect();
      const tr = toEl.getBoundingClientRect();
      const x1 = fr.left - surfaceRect.left + fr.width / 2;
      const y1 = fr.top - surfaceRect.top + fr.height / 2 + surface.scrollTop;
      const x2 = tr.left - surfaceRect.left + tr.width / 2;
      const y2 = tr.top - surfaceRect.top + tr.height / 2 + surface.scrollTop;
      // Control point: horizontal midpoint, vertical biased to introduce a curve
      const mx = (x1 + x2) / 2;
      const dy = Math.abs(y2 - y1);
      const cy = (y1 + y2) / 2;
      const d = `M ${x1.toFixed(1)} ${y1.toFixed(1)} Q ${mx.toFixed(1)} ${(cy - Math.min(60, dy * 0.3)).toFixed(1)} ${x2.toFixed(1)} ${y2.toFixed(1)}`;

      const kind = c.kind || "send";
      const path = document.createElementNS(svgNS, "path");
      path.setAttribute("class", "connection " + kind);
      path.setAttribute("d", d);
      path.setAttribute("marker-end", `url(#arrow-${kind})`);
      overlay.appendChild(path);

      if (c.label) {
        const tx = (x1 + 2 * mx + x2) / 4;
        const ty = (y1 + 2 * (cy - Math.min(60, dy * 0.3)) + y2) / 4;
        const bg = document.createElementNS(svgNS, "rect");
        const fg = document.createElementNS(svgNS, "text");
        fg.setAttribute("class", "connection-label " + kind);
        fg.setAttribute("x", String(tx));
        fg.setAttribute("y", String(ty));
        fg.setAttribute("text-anchor", "middle");
        fg.setAttribute("dominant-baseline", "middle");
        fg.textContent = c.label;
        // background rect sized after text is appended (we'll use bbox after the fact)
        overlay.appendChild(fg);
        try {
          const bbox = fg.getBBox ? fg.getBBox() : null;
          if (bbox && bbox.width) {
            const pad = 3;
            bg.setAttribute("class", "connection-label-bg");
            bg.setAttribute("x", String(bbox.x - pad));
            bg.setAttribute("y", String(bbox.y - pad));
            bg.setAttribute("width", String(bbox.width + pad * 2));
            bg.setAttribute("height", String(bbox.height + pad * 2));
            overlay.insertBefore(bg, fg);
          }
        } catch (_) { /* jsdom may not implement getBBox */ }
      }
    });
  }

  function ensureSvgDefs(overlay, svgNS) {
    if (overlay.querySelector("defs")) return;
    const defs = document.createElementNS(svgNS, "defs");
    const kinds = [
      { id: "send", color: "#3fb950" },
      { id: "receive", color: "#f0883e" },
      { id: "lookup", color: "#58a6ff" },
    ];
    kinds.forEach((k) => {
      const marker = document.createElementNS(svgNS, "marker");
      marker.setAttribute("id", "arrow-" + k.id);
      marker.setAttribute("viewBox", "0 0 10 10");
      marker.setAttribute("refX", "9");
      marker.setAttribute("refY", "5");
      marker.setAttribute("markerWidth", "6");
      marker.setAttribute("markerHeight", "6");
      marker.setAttribute("orient", "auto-start-reverse");
      const p = document.createElementNS(svgNS, "path");
      p.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
      p.setAttribute("fill", k.color);
      marker.appendChild(p);
      defs.appendChild(marker);
    });
    overlay.appendChild(defs);
  }

  // Debounced resize handler — re-draw current scene's connections.
  let _resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(_resizeTimer);
    _resizeTimer = setTimeout(() => {
      if (currentStepId && currentSceneKey !== undefined) {
        applyScene(currentStepId, currentSceneKey);
      }
    }, 120);
  });

  let currentSceneKey = null;
  let pinnedSceneKey = null;
  let pinnedRefEl = null;

  // --- AIR card ---
  function renderAirCard(air, opts) {
    opts = opts || {};
    const highlightSet = new Set(opts.highlightColumns || []);
    const card = el("div", { class: "air-viz", "data-air-id": air.id });

    // Header
    const header = el("div", { class: "air-viz-header" }, [
      el("span", { class: "air-name" }, [
        el("span", { class: "badge m-" + air.module, text: MODULE_LABEL[air.module] || air.module }),
        " " + air.name,
      ]),
      !opts.compact ? el("span", { class: "air-role", text: air.role || "" }) : null,
    ]);
    card.appendChild(header);

    // Trace table
    const wrap = el("div", { class: "air-table-wrap" });
    const table = el("table", { class: "air-trace" });
    const thead = el("thead");
    const tr = el("tr");
    if (!air.columns.length) {
      tr.appendChild(el("th", { text: "(no columns documented)" }));
    } else {
      air.columns.forEach((c) => {
        const th = el("th", {
          "data-col-name": c.name,
          text: c.name,
          title: `${c.name}${c.type ? " : " + c.type : ""}${c.description ? " — " + c.description : ""}`,
        });
        if (highlightSet.has(c.name)) th.classList.add("column-highlight");
        th.addEventListener("mouseenter", (ev) => {
          showTooltip(tipForColumn(c.name) + (c.description ? `<div class="tip-sub">${esc(c.description)}</div>` : ""), ev.clientX, ev.clientY);
        });
        th.addEventListener("mousemove", (ev) => {
          tipEl.style.left = ev.clientX + "px";
          tipEl.style.top = ev.clientY + "px";
        });
        th.addEventListener("mouseleave", hideTooltip);
        tr.appendChild(th);
      });
    }
    thead.appendChild(tr);
    table.appendChild(thead);

    // A few placeholder rows (representing generic trace rows)
    const tbody = el("tbody");
    const rowCount = opts.compact ? 2 : 3;
    const cols = Math.max(air.columns.length, 1);
    for (let r = 0; r < rowCount; r++) {
      const body = el("tr");
      for (let c = 0; c < cols; c++) {
        body.appendChild(el("td", { text: "···" }));
      }
      tbody.appendChild(body);
    }
    table.appendChild(tbody);
    wrap.appendChild(table);
    card.appendChild(wrap);

    // Bus chips
    if (air.buses && air.buses.length) {
      const buses = el("div", { class: "air-buses" });
      air.buses.forEach((b) => {
        const chip = el("span", {
          class: "bus-chip",
          "data-bus-name": b.name,
          "data-type": "bus",
          title: (busesByName[b.name] && busesByName[b.name].purpose) || "",
        }, [
          el("span", { class: "dir " + b.direction, text: b.direction }),
          b.name,
        ]);
        chip.addEventListener("mouseenter", (ev) => {
          const bus = busesByName[b.name];
          if (bus) showTooltip(tipForBus(bus), ev.clientX, ev.clientY);
          refHighlight("bus", b.name, true);
        });
        chip.addEventListener("mousemove", (ev) => {
          tipEl.style.left = ev.clientX + "px";
          tipEl.style.top = ev.clientY + "px";
        });
        chip.addEventListener("mouseleave", () => {
          hideTooltip();
          refHighlight("bus", b.name, false);
        });
        chip.addEventListener("click", () => {
          showSection("buses");
          // Scroll to bus
          setTimeout(() => {
            const target = document.querySelector(`.bus-full[data-bus-name="${b.name}"]`);
            if (target) target.scrollIntoView({ behavior: "smooth", block: "center" });
          }, 50);
        });
        buses.appendChild(chip);
      });
      card.appendChild(buses);
    }
    return card;
  }

  // --- Bus mini card (viz panel) ---
  function renderBusMini(bus) {
    const card = el("div", { class: "bus-card", "data-bus-name": bus.name });
    card.appendChild(
      el("div", { class: "bus-card-header" }, [
        el("span", { text: bus.name }),
        el("span", { class: "kind-pill " + bus.kind, text: bus.kind }),
      ])
    );
    const body = el("div", { class: "bus-card-body" });
    if (bus.purpose) body.appendChild(el("div", { class: "small muted", html: autoLink(bus.purpose) }));
    if (bus.producers && bus.producers.length) {
      body.appendChild(el("div", { class: "dir-label", text: "Producers" }));
      const prod = el("div", { class: "producers" });
      bus.producers.forEach((p) => {
        const air = airsByName[p.air] ? airsByName[p.air][0] : null;
        const peer = el("span", {
          class: "peer",
          "data-air-id": air ? air.id : p.air,
          text: `${p.direction} · ${p.air}`,
        });
        peer.addEventListener("mouseenter", () => refHighlight("air", air ? air.id : null, true));
        peer.addEventListener("mouseleave", () => refHighlight("air", air ? air.id : null, false));
        prod.appendChild(peer);
      });
      body.appendChild(prod);
    }
    if (bus.consumers && bus.consumers.length) {
      body.appendChild(el("div", { class: "dir-label", text: "Consumers" }));
      const cons = el("div", { class: "producers" });
      bus.consumers.forEach((p) => {
        const air = airsByName[p.air] ? airsByName[p.air][0] : null;
        const peer = el("span", {
          class: "peer",
          "data-air-id": air ? air.id : p.air,
          text: `${p.direction} · ${p.air}`,
        });
        peer.addEventListener("mouseenter", () => refHighlight("air", air ? air.id : null, true));
        peer.addEventListener("mouseleave", () => refHighlight("air", air ? air.id : null, false));
        cons.appendChild(peer);
      });
      body.appendChild(cons);
    }
    card.appendChild(body);
    return card;
  }

  // --- AIR browser ---
  let airFilterModule = "all";
  function renderAirFilter() {
    const container = $("air-filter");
    container.innerHTML = "";
    const all = el("button", { class: "filter-chip active", text: `All (${D.airs.length})` });
    all.addEventListener("click", () => filterAirsByModule("all"));
    container.appendChild(all);
    MODULE_ORDER.forEach((m) => {
      const count = D.airs.filter((a) => a.module === m).length;
      if (!count) return;
      const chip = el("button", { class: "filter-chip", text: `${MODULE_LABEL[m]} (${count})` }, [
        // trailing badge dot
      ]);
      chip.dataset.module = m;
      chip.addEventListener("click", () => filterAirsByModule(m));
      container.appendChild(chip);
    });
  }
  function filterAirsByModule(mod) {
    airFilterModule = mod;
    document.querySelectorAll("#air-filter .filter-chip").forEach((c) => {
      c.classList.toggle("active", (c.dataset.module || "all") === mod);
    });
    renderAirsGrid();
  }
  function renderAirsGrid() {
    const grid = $("airs-grid");
    grid.innerHTML = "";
    let airs = D.airs;
    if (airFilterModule !== "all") airs = airs.filter((a) => a.module === airFilterModule);
    // Group by module
    const byMod = {};
    airs.forEach((a) => (byMod[a.module] = byMod[a.module] || []).push(a));
    MODULE_ORDER.forEach((m) => {
      if (!byMod[m]) return;
      const header = el("h3", {}, [
        el("span", { class: "badge m-" + m, text: MODULE_LABEL[m] }),
        " " + MODULE_LABEL[m] + " — " + byMod[m].length + " AIRs",
      ]);
      grid.appendChild(header);
      byMod[m].forEach((a) => grid.appendChild(renderAirCard(a, { compact: false })));
    });
  }

  // --- Bus browser ---
  function renderBusBrowser() {
    const grid = $("buses-grid");
    grid.innerHTML = "";
    const sorted = [...D.buses].sort((a, b) => {
      if (a.kind !== b.kind) return a.kind === "permutation" ? -1 : 1;
      return a.name.localeCompare(b.name);
    });
    sorted.forEach((bus) => grid.appendChild(renderBusFull(bus)));
  }
  function renderBusFull(bus) {
    const card = el("div", { class: "bus-full", "data-bus-name": bus.name });
    card.appendChild(
      el("h3", {}, [
        el("span", { class: "mono", text: bus.name }),
        el("span", { class: "kind-pill " + bus.kind, text: bus.kind }),
      ])
    );
    if (bus.purpose) card.appendChild(el("div", { class: "purpose small", html: autoLink(bus.purpose) }));
    if (bus.message_format && bus.message_format.length) {
      const tbl = el("table", { class: "msg-format" });
      const th = el("tr");
      th.appendChild(el("th", { text: "field" }));
      th.appendChild(el("th", { text: "type" }));
      th.appendChild(el("th", { text: "description" }));
      tbl.appendChild(th);
      bus.message_format.forEach((f) => {
        const tr = el("tr");
        tr.appendChild(el("td", { text: f.field }));
        tr.appendChild(el("td", { text: f.type || "?" }));
        tr.appendChild(el("td", { text: f.description || "" }));
        tbl.appendChild(tr);
      });
      card.appendChild(tbl);
    }

    function renderPeers(label, list) {
      if (!list || !list.length) return;
      card.appendChild(el("div", { class: "dir-label small", text: label }));
      const plist = el("div", { class: "peer-list" });
      list.forEach((p) => {
        const air = airsByName[p.air] ? airsByName[p.air][0] : null;
        const chip = el("span", {
          class: "peer",
          "data-air-id": air ? air.id : p.air,
          title: p.note || "",
          text: `${p.direction} · ${p.air}`,
        });
        chip.addEventListener("click", () => {
          if (air) {
            filterAirsByModule(air.module);
            showSection("airs");
            setTimeout(() => {
              const target = document.querySelector(`[data-air-id="${air.id}"]`);
              if (target) target.scrollIntoView({ behavior: "smooth", block: "center" });
            }, 50);
          }
        });
        chip.addEventListener("mouseenter", () => refHighlight("air", air ? air.id : null, true));
        chip.addEventListener("mouseleave", () => refHighlight("air", air ? air.id : null, false));
        plist.appendChild(chip);
      });
      card.appendChild(plist);
    }
    renderPeers("Producers (Send / Provide)", bus.producers);
    renderPeers("Consumers (Receive / Lookup)", bus.consumers);

    if (bus.invariants) {
      card.appendChild(
        el("div", { class: "small dim", style: "margin-top:10px", html: "<strong>Invariant:</strong> " + autoLink(bus.invariants) })
      );
    }
    return card;
  }

  // --- Correctness concerns ---
  function renderConcerns() {
    const container = $("concerns-list");
    container.innerHTML = "";
    (D.concerns || []).forEach((c) => {
      const card = el("div", { class: "ov-card", style: "margin-bottom:14px" });
      card.appendChild(el("h3", { style: "margin-top: 4px", text: c.title }));
      card.appendChild(
        el("div", { html: `<div class="section-label">Concern</div><div>${autoLink(c.concern || "")}</div>` })
      );
      card.appendChild(
        el("div", { style: "margin-top: 12px", html: `<div class="section-label">Resolution</div><div>${autoLink(c.resolution || "")}</div>` })
      );
      const parts = [];
      if (c.key_airs && c.key_airs.length)
        parts.push(
          "<strong>Key AIRs:</strong> " +
            c.key_airs
              .map((aid) => {
                const a = airsById[aid];
                const name = a ? a.name : aid;
                return `<span class="ref" data-type="air" data-id="${esc(aid)}">${esc(name)}</span>`;
              })
              .join(", ")
        );
      if (c.key_buses && c.key_buses.length)
        parts.push(
          "<strong>Key buses:</strong> " +
            c.key_buses.map((b) => `<span class="ref" data-type="bus" data-id="${esc(b)}">${esc(b)}</span>`).join(", ")
        );
      if (parts.length)
        card.appendChild(el("div", { class: "small muted", style: "margin-top:12px", html: parts.join(" &nbsp;·&nbsp; ") }));
      container.appendChild(card);
    });
  }

  // --- Global hover/click highlighting ---
  //
  // Scene-triggering elements come in two flavours:
  //   1) .ref[data-scene]        — fine-grained refs inside prose (legacy)
  //   2) .code-block[data-scene] — the primary trigger in chapter mode
  //
  // Both are treated the same way: hover switches the scene, click pins it.
  // For .ref elements without data-scene we still show a tooltip and do a
  // coarse highlight of the AIR/bus/column.

  function sceneTarget(t) {
    if (!t) return null;
    if (t.classList && t.classList.contains("ref") && t.dataset.scene) return t;
    if (t.classList && t.classList.contains("code-block") && t.dataset.scene) return t;
    // For code-block, the mouse may land on an inner child; walk up.
    if (t.closest) {
      const cb = t.closest(".code-block[data-scene]");
      if (cb) return cb;
    }
    return null;
  }

  document.addEventListener("mouseenter", (ev) => {
    const t = ev.target;
    if (!t || !t.classList) return;
    // Code-block hover: bubbles up via the capture phase — handle only when
    // the entered element IS a code-block, not some inner child (otherwise
    // the activation fires repeatedly as the mouse moves over inner nodes).
    if (t.classList.contains("code-block") && t.dataset.scene) {
      activateSceneBlock(t, false);
      return;
    }
    if (!t.classList.contains("ref")) return;
    const sceneKey = t.dataset.scene;
    const type = t.dataset.type;
    const id = t.dataset.id;
    t.classList.add("active");
    if (sceneKey) {
      if (pinnedSceneKey) return;
      currentSceneKey = sceneKey;
      currentStepId = sceneKey;
      renderStepViz({ id: sceneKey, airs: [], buses: [] });
      return;
    }
    refHighlight(type, id, true);
    if (type === "air") {
      const air = airsById[id];
      if (air) showTooltip(tipForAir(air), ev.clientX, ev.clientY);
    } else if (type === "bus") {
      const bus = busesByName[id];
      if (bus) showTooltip(tipForBus(bus), ev.clientX, ev.clientY);
    } else if (type === "col") {
      showTooltip(tipForColumn(id), ev.clientX, ev.clientY);
    }
  }, true);

  document.addEventListener("mousemove", (ev) => {
    const t = ev.target;
    if (t && t.classList && t.classList.contains("ref")) {
      tipEl.style.left = ev.clientX + "px";
      tipEl.style.top = ev.clientY + "px";
    }
  }, true);

  document.addEventListener("mouseleave", (ev) => {
    const t = ev.target;
    if (!t || !t.classList) return;
    if (t.classList.contains("code-block") && t.dataset.scene) {
      // On code-block leave: keep the viz latched. We only remove the .active
      // affordance; the scene stays on screen until another block is hovered
      // or the user explicitly unpins.
      // Rationale: with a long scrollable chapter, clearing the viz on each
      // leave would make it almost always empty.
      return;
    }
    if (!t.classList.contains("ref")) return;
    const sceneKey = t.dataset.scene;
    const type = t.dataset.type;
    const id = t.dataset.id;
    if (!(sceneKey && pinnedRefEl === t)) t.classList.remove("active");
    if (sceneKey) {
      if (pinnedSceneKey) return;
      // Same "latch" behavior as code blocks — keep the viz on the last scene.
      return;
    }
    refHighlight(type, id, false);
    hideTooltip();
  }, true);

  // Click-to-pin: clicking a scene-trigger (either .code-block[data-scene] or
  // .ref[data-scene]) pins that scene; clicking the same trigger again unpins;
  // clicking anywhere outside any trigger also unpins.
  document.addEventListener("click", (ev) => {
    const t = ev.target && ev.target.closest
      ? ev.target.closest(".code-block[data-scene], .ref[data-scene]")
      : null;
    if (t) {
      const sceneKey = t.dataset.scene;
      if (pinnedRefEl === t) {
        // Toggle off.
        if (pinnedRefEl) pinnedRefEl.classList.remove("pinned");
        pinnedSceneKey = null;
        pinnedRefEl = null;
        currentSceneKey = null;
        // Keep the latched viz (no revert). User can hover another to change.
      } else {
        if (pinnedRefEl) pinnedRefEl.classList.remove("pinned");
        pinnedSceneKey = sceneKey;
        pinnedRefEl = t;
        t.classList.add("pinned");
        currentSceneKey = sceneKey;
        currentStepId = sceneKey;
        renderStepViz({ id: sceneKey, airs: [], buses: [] });
      }
      ev.stopPropagation();
      return;
    }
    // Click outside any scene trigger: unpin (but keep viz latched).
    if (pinnedSceneKey) {
      if (pinnedRefEl) pinnedRefEl.classList.remove("pinned");
      pinnedSceneKey = null;
      pinnedRefEl = null;
    }
  });

  function refHighlight(type, id, on) {
    if (!id) return;
    const cls = on ? "add" : "remove";
    if (type === "air") {
      document.querySelectorAll(`[data-air-id="${cssEscape(id)}"]`).forEach((e) => e.classList[cls]("highlight"));
    } else if (type === "bus") {
      document.querySelectorAll(`[data-bus-name="${cssEscape(id)}"]`).forEach((e) => e.classList[cls]("highlight"));
      // Also highlight AIR cards currently rendered that connect to this bus
      const bus = busesByName[id];
      if (bus) {
        (bus._airSet || new Set()).forEach((airName) => {
          const a = airsByName[airName] ? airsByName[airName][0] : null;
          if (a) {
            document.querySelectorAll(`.air-viz[data-air-id="${cssEscape(a.id)}"]`).forEach((e) => e.classList[cls]("highlight"));
          }
        });
      }
    } else if (type === "col") {
      document.querySelectorAll(`th[data-col-name="${cssEscape(id)}"]`).forEach((e) => e.classList[cls]("column-highlight"));
    }
  }

  // --- Boot ---
  renderOverview();
  renderPhaseTabs();
  selectChapter(CHAPTER_ORDER[0]);
  renderAirFilter();
  renderAirsGrid();
  renderBusBrowser();
  renderConcerns();

  // Optional deep-link support via hash.
  //   #phase=<id>   — open a specific phase page
  //   #scene=<key>  — open the phase containing that scene key and activate it
  //   #air=<id>     — open AIR browser filtered to that module
  //   #bus=<name>   — open Bus browser
  //   #overview|proof|airs|buses|concerns — jump directly to a section
  const hash = location.hash.replace(/^#/, "");
  if (hash) {
    const [k, v] = hash.split("=");
    if (k === "phase" && v) {
      selectChapter(v);
      showSection("proof");
    } else if (k === "scene" && v) {
      const phaseForScene = CHAPTER_ORDER.find((pid) => {
        const ch = getChapter(pid);
        return ch && (ch.blocks || []).some((b) => b.scene === v);
      });
      if (phaseForScene) selectChapter(phaseForScene);
      showSection("proof");
      setTimeout(() => {
        const cb = document.querySelector(
          `.code-block[data-scene="${cssEscape(v)}"]`
        );
        if (cb) {
          cb.scrollIntoView({ behavior: "smooth", block: "center" });
          activateSceneBlock(cb, false);
        }
      }, 60);
    } else if (k === "step" && v) {
      // Legacy deep-link: map stepId → scene key (they're the same).
      const phaseForScene = CHAPTER_ORDER.find((pid) => {
        const ch = getChapter(pid);
        return ch && (ch.blocks || []).some((b) => b.scene === v);
      });
      if (phaseForScene) selectChapter(phaseForScene);
      showSection("proof");
    } else if (k === "air") {
      const a = airsById[v];
      if (a) {
        filterAirsByModule(a.module);
        showSection("airs");
      }
    } else if (k === "bus") {
      showSection("buses");
    } else if (k === "overview" || k === "proof" || k === "airs" || k === "buses" || k === "concerns") {
      showSection(k);
    }
  }
})();
