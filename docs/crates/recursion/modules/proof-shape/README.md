# ProofShape Module

**Reference verifier correspondence:** the preamble of `verify` (`verifier/mod.rs`, lines
126-155) — VK pre-hash observation, commitment observations, public value observations,
sort order, and derived quantities (`n_logup`, `n_max`, etc.).

**AIRs (3):** ProofShapeAir, PublicValuesAir, RangeCheckerAir.
**Per-AIR details:** [ProofShape AIRs](./airs.md).

**Internal bus:** `NumPublicValuesBus` (perm) — ProofShapeAir sends `(air_idx, tidx, num_pvs)`
to PublicValuesAir, coordinating how many public values each AIR contributes.

## Interface

ProofShapeAir is the data-flow origin — no incoming module handoff messages. Its outputs
feed every downstream module.

### VK constants (baked into circuit)

- `l_skip` — log-blowup / skip factor
- `max_interaction_count` — upper bound on total interactions
- Per-AIR metadata: `is_required`, `air_idx`, `num_interactions`, `need_rot`,
  preprocessed commits, trace partition widths, `num_public_values`

### Outputs

Per-AIR (via buses listed in PS2 below): structural data, presence flags, heights,
commitments, hypercube dimensions, lift dimensions.

Per-proof: `(tidx_end, n_logup, n_max, is_n_max_greater)` on `GkrModuleBus` to start the
pipeline; `(n_logup, n_max)` on `EqNsNLogupMaxBus`; `(n_max)` on `ExpressionClaimNMaxBus`;
`(num_present_airs)` on `FractionFolderInputBus`.

Public values: `(air_idx, pv_idx, value)` on `PublicValuesBus` (via PublicValuesAir).

VK pre-hash: sent on `PreHashBus` when continuations are enabled (see
[bus-inventory.md](../../bus-inventory.md)). Always observed on the transcript.

## Sort order

The reference verifier sorts all `num_airs` AIRs by the key
`(is_none, Reverse(log_height), air_id)`. ProofShapeAir has one row per AIR (`num_airs`
rows total), with `sort_idx` ranging from 0 to `num_airs - 1`.

ProofShapeAir constrains the sort order via:
- **Non-increasing heights:** `log_height[i] - log_height[i+1] >= 0` (range-checked via
  `RangeCheckerBus`).
- **Absent AIRs last:** `log_height = 0` and `height = 0` when `is_present = 0`.
- **Permutation:** a permutation bus enforces that the `sort_idx → air_idx` mapping is a
  bijection (every AIR appears exactly once).

The circuit does **not** constrain the tiebreaker within a height tie — the ordering of
AIRs with equal `log_height` is a prover choice. This is sound because:

- **Internal consistency.** Every downstream use of `sort_idx` receives both structural
  metadata (air_idx, num_interactions, need_rot via `AirShapeBus`) and proof data (column
  openings, heights via `LiftedHeightsBus`, etc.) through the same buses, keyed by the same
  `sort_idx`. Reordering within a tie relabels all data consistently.

- **Verifier invariance.** The reference verifier's algorithm only relies on trace heights
  being in non-increasing order — it does not depend on which specific AIR occupies which
  sort position within a height tie. The extraction produces `(trace_vdata, proof_arrays)`
  indexed by the circuit's sort order; the reference verifier sorts `trace_vdata` (producing
  *some* non-increasing-height permutation) and iterates. Any such permutation leads to the
  same verification result.

## Empty traces

The reference verifier rejects proofs with zero present AIRs (`num_traces == 0`). In the
circuit, this is guaranteed by PS1b: at least one AIR is VK-required, so at least one
AIR must be present. (A VK with zero required AIRs would fail at circuit construction time.)

## Extraction

Let `P` denote the extracted proof for a given `proof_idx`. Let `i` range over AIRs in sort
order (i.e. `i = sort_idx`). ProofShape populates:

Per-AIR (indexed by `(proof_idx, i)`):
- `P.is_present[i]` — whether AIR `i` is present
- `P.log_height[i]` — encodes `log₂(height)` when `P.is_present[i] = 1`; canonical dummy
  value `0` when absent
- `P.cached_commits[i][0..max_cached]` — cached trace commitments

Per-proof (indexed by `proof_idx`):
- `P.common_main_commit` — main trace commitment (DIGEST_SIZE elements)
- `P.is_n_max_greater` — boolean flag (see PS1c)
- `P.public_values[air_idx][pv_idx]` — public value array (via PublicValuesAir)

These are the prover-supplied degrees of freedom. Everything else ProofShapeAir outputs
is either a VK constant or derived from the above by constrained arithmetic.

### Derived quantities and conventions

- `height[i] := 0` when `P.is_present[i] = 0`; `2^P.log_height[i]` when `P.is_present[i] = 1`
- `n[i] := P.log_height[i] - l_skip` (signed)
- `n_abs[i] := |n[i]|`, `n_sign_bit[i] := (n[i] < 0 ? 1 : 0)`
- `lifted_height[i] := 0` when not present; `max(height[i], 2^l_skip)` when present
- `log_lifted_height[i] := 0` when not present; `max(P.log_height[i], l_skip)` when present
- `n_lift[i] := max(0, n[i])`
- `total_interactions := sum_i(num_interactions_per_row[i] (VK) * lifted_height[i])`
- `n_logup` := index of the most significant set bit of `total_interactions`; 0 when
  `total_interactions = 0`
- `n_max` := max `n_lift[i]` across present AIRs (the set is nonempty since at least one
  AIR is VK-required)
- `num_present_airs` := count of `i` with `P.is_present[i] = 1`
- `tidx_end` := first transcript index after the complete ProofShape/PublicValues preamble
  for this proof

For absent AIRs (`P.is_present[i] = 0`), the derived quantities are arithmetic dummy values.
All semantic uses in downstream modules are gated by presence.

## Statement

**(PS1a) Witness well-formedness.**
- `P.is_present[i]` is boolean.
- `P.log_height[i]` is range-checked (via `PowerCheckerBus`, which simultaneously
  enforces the power-of-two relation).
- `height[i] = 2^P.log_height[i]` when present (via `PowerCheckerBus`).

**(PS1b) VK compatibility.**
- VK-required AIRs have `P.is_present[i] = 1`.
- Per-AIR VK metadata (`air_idx`, `num_interactions`, `need_rot`, preprocessed commits,
  widths) used in bus outputs and derived quantities is fixed by `vk_child`.

**(PS1c) Derived global quantities.**
- `total_interactions < max_interaction_count` (VK constant).
- `P.is_n_max_greater`: 1 if `n_max > n_logup`, 0 if `n_max < n_logup`, feasibly either
  when `n_max = n_logup`. This ambiguity is harmless — see [GKR](../gkr/README.md) (note on
  `is_n_max_greater`).

**(PS2) Bus outputs.** Per present AIR (indexed by `(proof_idx, i)`, sent only when
`is_present = 1` unless noted):

| Bus | Payload |
| --- | --- |
| `AirShapeBus` | `(i, air_idx (VK), num_interactions (VK), need_rot (VK))` |
| `AirPresenceBus` | `(air_idx (VK), P.is_present[i])` — sent for **all** AIRs (present and absent) |
| `HyperdimBus` | `(i, n_abs[i], n_sign_bit[i])` |
| `LiftedHeightsBus` | `(i, part_idx, commit_idx, n[i], lifted_height[i], log_lifted_height[i])` — one message per trace partition (main, preprocessed, each cached) |
| `CommitmentsBus` | `(major_idx, minor_idx, commitment)` — the `(major_idx, minor_idx)` pair determines whether the commitment is VK-fixed (preprocessed) or witness-supplied (`P.cached_commits[i]`) |
| `NLiftBus` | `(air_idx (VK), n_lift[i])` |
| `Eq3bShapeBus` | `(i, n_lift[i], n_logup, num_interactions[i])` |

Note: `AirShapeBus` carries only VK data. It is a VK broadcast that downstream modules use
to look up structural constants by `sort_idx`.

Per-proof (indexed by `proof_idx`):

| Bus | Payload |
| --- | --- |
| `GkrModuleBus` | `(tidx_end, n_logup, n_max, P.is_n_max_greater)` |
| `EqNsNLogupMaxBus` | `(n_logup, n_max)` |
| `ExpressionClaimNMaxBus` | `(n_max)` |
| `FractionFolderInputBus` | `(num_present_airs)` |

**(PS3) Transcript preamble.** ProofShapeAir observes (via `TranscriptBus`) in order:

```
observe vk_pre_hash                                        (DIGEST_SIZE)
observe common_main_commit                                 (DIGEST_SIZE)
for each AIR i in air_idx order:
  observe is_present[i]                                    (if not VK-required)
  observe preprocessed_commit[i]                           (DIGEST_SIZE, if applicable, VK)
  observe log_height[i]                                    (if no preprocessed data)
  observe cached_commits[i]                                (DIGEST_SIZE each)
observe public_values                                      (via PublicValuesAir)
```

Note: ProofShapeAir rows are in sort_idx order (descending height), but each row carries
a precomputed `starting_tidx` from the preflight that corresponds to the air_idx-order
position. The transcript indices thus match the reference verifier's air_idx iteration.

These observations occupy a contiguous transcript interval beginning at `tidx = 0`;
`tidx_end` is the first unused index after this interval. The sequence of observed values
agrees with the preamble observations performed by the reference `verify`.

The VK pre-hash is not constrained to equal `hash(vk_child)` within the subcircuit;
see the provisos on the main claim.

**(PS4) PublicValuesAir.** PublicValuesAir has one row per public value across all AIRs.
Each row contains `(proof_idx, air_idx, pv_idx, tidx, value)`. For each row:
- **Transcript observation:** Receives from `TranscriptBus` at the `tidx` assigned by PS3,
  with `is_sample = 0` and `value`. This pins the row's `value` to the
  transcript-observed value in the preamble (matching the reference verifier's
  `transcript.observe(*pv)` loop).
- **PublicValuesBus send:** Sends `(air_idx, pv_idx, value)` on `PublicValuesBus`.
  BatchConstraint's SymbolicExpressionAir receives these and uses them when evaluating
  `Entry::Public` variables in the constraint DAG (see [BatchConstraint](../batch-constraint/README.md),
  step (a)).
- **NumPublicValuesBus receive:** Receives `(air_idx, tidx, num_pvs)` from ProofShapeAir
  to determine how many PVs each AIR contributes and at what `tidx` they start.

Because the same `value` cell is used for both the `TranscriptBus` receive and the
`PublicValuesBus` send, the extracted `P.public_values` are guaranteed to be exactly the
values observed on the transcript and the values used in constraint evaluation. This mirrors
the reference verifier, which observes `public_values` on the transcript (preamble) and then
passes the same vector to `verify_zerocheck_and_logup` for constraint evaluation.

**Lookup dependencies:** `PowerCheckerBus`, `RangeCheckerBus` (self-provided).

## Module-level argument

ProofShape corresponds to the preamble of the reference verifier's `verify()` (lines 126–155)
plus the derived-quantity computations (sorting, `n_per_trace`, `total_interactions` check).

1. **Transcript correspondence.** PS3 establishes that ProofShapeAir and PublicValuesAir
   observe on `TranscriptBus` at the same `tidx` positions and in the same order as the
   reference verifier's preamble loop. The preamble iterates in air_idx order; the circuit
   matches this via precomputed `starting_tidx` values.

2. **Verification correspondence.** PS1a–PS1c establish that the extracted proof attributes
   and derived quantities satisfy the same well-formedness, VK-compatibility, and bound
   checks that the reference verifier enforces (sort order, `verify_proof_shape` checks,
   trace height constraints). PS2 establishes that the bus outputs are deterministic
   functions of the extracted attributes and VK constants, providing correct inputs to
   downstream modules. PS4 establishes that public values are correctly threaded from the
   transcript to constraint evaluation.

Together, these guarantee that the ProofShape extraction produces a valid preamble for the
reference verifier, and that the outgoing bus messages (`GkrModuleBus`, structural data
buses, `PublicValuesBus`) carry the correct values for downstream modules.
