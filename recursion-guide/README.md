# OpenVM Recursion Sub-Circuit — Interactive Guide

A standalone static website that walks through the proof of why the OpenVM
recursion sub-circuit correctly constrains the stark-backend `verify()` function.

## Viewing

Open `index.html` directly in a browser — no build step, no server needed.

```bash
# macOS
open index.html

# Linux
xdg-open index.html

# Or just drag index.html onto a browser tab.
```

Everything runs from `file://`; data is embedded via `js/data.js`.

## Sections

- **Overview** — main correspondence claim, assumptions, decomposition into sub-claims (T, PS, LT, M-GKR, M-BC, M-Stack, M-WHIR), plus the architecture diagram.
- **Proof Walkthrough** — the central content, now organized as 5 phase pages (Preamble, GKR, Batch Constraint, Stacking, WHIR). Each page presents the Rust `Air::eval` implementations in source order with prose explanations between code chunks. Hovering a code chunk highlights the cells it constrains and draws the bus connections it drives in the right-hand panel; clicking pins it.
- **AIR Browser** — all 39 AIRs as trace tables with labeled columns. Filterable by module.
- **Bus Browser** — all 80 buses with message formats and producer/consumer lists.
- **Correctness Concerns** — transcript ordering, LogUp soundness, bus completeness, boundary conditions, challenge independence.

## Deep links

- `#phase=gkr` — open a specific phase page
- `#scene=bc-11` — open the phase containing that scene key and scroll/activate its code block
- `#step=gkr-3` — legacy alias for `#scene=gkr-3`
- `#air=whir.SumcheckAir` — open the AIR browser filtered to that AIR's module
- `#buses` — jump directly to the bus browser

## Data sources

All content derives from the docs under `docs/crates/recursion/`:

- `AIR_MAP.md` — AIR inventory and bus connectivity
- `bus-inventory.md` — bus message formats, invariants, peers
- `verifier-mapping.md` — step-by-step correspondence to `verify()`
- `modules/*/README.md` — module-level arguments
- `modules/*/airs.md` — per-AIR trace columns

Data was extracted into `data/airs.json`, `data/buses.json`, `data/walkthrough.json`, and bundled as `js/data.js` for file:// compatibility.

## Regenerating

After updating the underlying docs, rebuild `js/data.js` by re-running the
extraction. `data/airs.json`, `data/buses.json`, `data/walkthrough.json`,
scene bundles under `data/scenes/*.json`, and chapter pages under
`data/chapters/*.json` are concatenated into `window.DATA`.

```bash
python3 -c "
import json, glob, os
a = json.load(open('data/airs.json'))
b = json.load(open('data/buses.json'))
w = json.load(open('data/walkthrough.json'))
scenes = {}
for p in sorted(glob.glob('data/scenes/*.json')):
  with open(p) as f: d = json.load(f)
  for k, v in d.items():
    if k in scenes:
      existing = scenes[k]
      merged = dict(existing)
      merged.update({k2: v2 for k2, v2 in v.items() if k2 != 'scenes'})
      merged['scenes'] = {**(existing.get('scenes') or {}), **(v.get('scenes') or {})}
      scenes[k] = merged
    else:
      scenes[k] = v
chapters = {}
for p in sorted(glob.glob('data/chapters/*.json')):
  with open(p) as f: d = json.load(f)
  pid = d.get('id') or os.path.splitext(os.path.basename(p))[0]
  chapters[pid] = d
with open('js/data.js','w') as f:
  f.write('/* Auto-generated */\n')
  f.write('window.DATA = {\n')
  f.write('  airs: ' + json.dumps(a['airs'],separators=(',',':')) + ',\n')
  f.write('  buses: ' + json.dumps(b['buses'],separators=(',',':')) + ',\n')
  f.write('  main_claim: ' + json.dumps(w['main_claim'],separators=(',',':')) + ',\n')
  f.write('  phases: ' + json.dumps(w['phases'],separators=(',',':')) + ',\n')
  f.write('  concerns: ' + json.dumps(w['correctness_concerns'],separators=(',',':')) + ',\n')
  f.write('  scenes: ' + json.dumps(scenes,separators=(',',':')) + ',\n')
  f.write('  chapters: ' + json.dumps(chapters,separators=(',',':')) + '\n')
  f.write('};\n')
"
```

### Chapter schema

Each `data/chapters/<phase>.json` is one scrollable page:

```
{
  "id": "<phase>",
  "name": "Phase N: <Title>",
  "summary": "<HTML>",
  "blocks": [
    { "type": "prose", "html": "<HTML>" },
    { "type": "air_header", "air_id": "<module>.<AirName>",
      "heading": "<label>", "html": "<HTML>", "source_path": "..." },
    { "type": "code", "lang": "rust",
      "heading": "<short title>",
      "source_path": "crates/recursion/src/.../air.rs",
      "source_lines": "lo-hi",
      "code": "<verbatim Rust source slice>",
      "explain": "<HTML explanation>",
      "scene": "<scene-key referencing data/scenes/*.json>"
    }
  ]
}
```

The `scene` field on a `code` block is the scene-bundle key (e.g.
`preamble-3`). Hovering the block activates its default scene; clicking
pins it. Blocks with no `scene` are still rendered (useful for struct
definitions or plumbing code) but don't drive the viz panel.

### Scene schema

Each `data/scenes/*.json` file is a map from `stepId` → scene bundle:

```
{
  "<step-id>": {
    "default_scene": Scene,
    "scenes": { "<scene-name>": Scene, ... },
    "constraints_html": "optional HTML — may contain <span class='ref' data-scene='X'>...</span>",
    "soundness_html": "optional HTML"
  }
}
```

Where `Scene` is:

```
{
  "airs":        [{ "name": "<air>", "rows": [{ "row": <int>, "cols": ["<col>"], "kind": "primary|prev|next|secondary" }] }],
  "buses":       [{ "name": "<bus>", "rows": [{ "row": <int>, "fields": ["<field>"], "kind": "primary|secondary" }] }],
  "connections": [{ "from": { "air": "...", "col": "...", "row": <int> },
                    "to":   { "bus": "...", "field": "...", "row": <int> },
                    "kind": "send|receive|lookup", "label": "..." }],
  "note": "<optional short caption>"
}
```

Row indices: `0, 1, 2` index the first three rows; `-1` means the last row ("n-1").
The renderer draws exactly 4 visible rows (0, 1, 2, n-1) with an ellipsis between.
Any row index outside {0, 1, 2, -1} (e.g. `row: 3`) clamps to the ellipsis marker.
