# GKR Module

**Reference verifier correspondence:** the GKR portion of `verify_zerocheck_and_logup`
(`verifier/batch_constraints.rs`) and `verify_gkr` (`verifier/fractional_sumcheck_gkr.rs`)
— logup PoW, alpha/beta sampling, `verify_gkr` (layer recurrence, sumcheck), xi extension.
The boundary with BatchConstraint is after `verify_gkr` returns and xi is assembled: GKR
handles everything up to and including xi extension; BatchConstraint handles constraint
evaluation, claim assembly, and the sumcheck that follows.

**AIRs (4):** GkrInputAir, GkrLayerAir, GkrLayerSumcheckAir, GkrXiSamplerAir.
**Per-AIR details:** [GKR AIRs](./airs.md).

## Interface

All quantities below are per proof (indexed by `proof_idx`). Bus names are in parentheses for
cross-reference with [bus-inventory.md](../../bus-inventory.md).

### VK constants (baked into circuit)

- `l_skip` — log-blowup / skip factor
- `pow_bits` — proof-of-work difficulty (0 means no PoW)

### Inputs

- **Pipeline handoff:** `tidx` — starting transcript index for this proof's GKR activity,
  equal to `tidx_end` from ProofShape (`GkrModuleBus`, received by GkrInputAir)
- **Proof shape:** `n_logup`, `n_max`, `is_n_max_greater` (`GkrModuleBus`, received by
  GkrInputAir)

Note on `is_n_max_greater`: ProofShapeAir guarantees `is_n_max_greater = 1` when
`n_max > n_logup` and `= 0` when `n_max < n_logup`, but when `n_max = n_logup` either value
is feasible at the ProofShape level. However, GKR requires `is_n_max_greater = 0` when
`n_max = n_logup > 0`: setting it to 1 enables the xi sampler bus interactions with
`num_challenges = 0`, leaving `GkrXiSamplerBus` messages unbalanced (GkrInputAir sends/receives
with mismatched indices that the dummy xi sampler cannot satisfy). When `n_logup = 0` the flag
is irrelevant because `needs_challenges` is forced true by `is_n_logup_zero`.

### Derived quantities

- `K = n_logup + l_skip` — total GKR layers (`total_rounds` in the reference verifier)
- `n_global = max(n_logup, n_max)`

### Outputs

- **GKR claim:** `tidx_bc` and `gkr_claim = [p_claim, q_claim - alpha_logup]` in D_EF —
  the transcript position and aggregated logup claim returned by `verify_gkr`, or `[0, 0]`
  when `n_logup = 0`. `tidx_bc = tidx + pow_offset + 2*D_EF +
  has_interactions * K*(K+2)*2*D_EF + needs_challenges * num_challenges * D_EF + D_EF`
  where `K = n_logup + l_skip`, `has_interactions = (n_logup > 0) ? 1 : 0`,
  `pow_offset = pow_tidx_count(pow_bits)`,
  `needs_challenges = (n_max > n_logup) || (n_logup == 0)`,
  `num_challenges = n_max + l_skip - has_interactions * K`
  (`BatchConstraintModuleBus`, sent by GkrInputAir)
- **Evaluation point:** `xi[0..n_global + l_skip]` in D_EF — assembled from the last GKR
  layer's challenges (schedule (4)–(5) below). (`XiRandomnessBus`, sent by three AIRs):
  - `idx = 0`: `mu_{K-1}` (last GKR layer's mu) — sent by GkrLayerAir
  - `idx = 1..K-1`: sumcheck challenges from the last layer's `K-1` rounds — sent by
    GkrLayerSumcheckAir
  - `idx = K..`: additional transcript-sampled challenges (present when
    `n_max > n_logup`, or when `n_logup = 0`) — sent by GkrXiSamplerAir
  - When `n_logup = 0`: all entries are transcript-sampled (GkrXiSamplerAir only)
- **Transcript positions for BatchConstraint** (both sent by GkrInputAir):
  - `tidx_cf` — transcript index after (5), where BC samples `lambda`
    (`ConstraintsFoldingInputBus`)
  - `tidx_if = tidx + pow_offset + D_EF` — transcript index right after (2), where BC
    samples `beta_logup` (`InteractionsFoldingInputBus`)

## Extraction

From the satisfying GKR traces, read off a candidate `GkrProof`:

| Extracted field | Source AIR | Description |
| --- | --- | --- |
| `logup_pow_witness` | GkrInputAir | PoW witness value |
| `q0_claim` | GkrInputAir | initial GKR claim (D_EF) |
| `claims_per_layer[j].{p_xi_0, q_xi_0, p_xi_1, q_xi_1}` | GkrLayerAir | per-layer boundary claims |
| `sumcheck_polys[j][i] = [ev1, ev2, ev3]` | GkrLayerSumcheckAir | per-layer per-round sumcheck evaluations |

When `n_logup = 0`, the extracted proof has empty `claims_per_layer` and `sumcheck_polys`.

Note: GkrXiSamplerAir does not contribute extracted proof fields — it only samples additional
xi coordinates from the transcript (schedule (5)) and sends them on `XiRandomnessBus`. These are
challenges, not prover-supplied data.

**Connection to outgoing buses.** GkrInputAir receives the last layer's
`(numer_claim, denom_claim)` from GkrLayerAir (via internal GkrLayerOutputBus) and assembles
`gkr_claim = [numer_claim, denom_claim - alpha_logup]`, which it sends on
`BatchConstraintModuleBus`. The `(numer_claim, denom_claim)` pair is the linear interpolation
of `claims_per_layer[K-1]` at the sampled `mu_{K-1}` (see `reduce_to_single_evaluation` in
`gkr/layer/air.rs`). GkrInputAir also sends `tidx_cf = tidx_bc - D_EF` on
`ConstraintsFoldingInputBus` and `tidx_if = tidx + pow_offset + D_EF` on
`InteractionsFoldingInputBus`.

## Contract

Execute the GKR-relevant slice of `verify_zerocheck_and_logup` with the extracted data.
The transcript schedule is:

```
(1) observe logup_pow_witness, sample logup_pow_sample   (if pow_bits > 0)
(2) sample alpha_logup                                   (D_EF)
    [gap: D_EF reserved for beta_logup, sampled by BC]
(3) observe q0_claim                                     (D_EF, if n_logup > 0)
(4) for layer j = 0, ..., K-1:                           (if n_logup > 0)
      sample lambda_j                                    (if j > 0)
      for sub-round i = 0, ..., j-1:                     (if j > 0)
        observe ev1, ev2, ev3
        sample r_i
      observe p_xi_0, q_xi_0, p_xi_1, q_xi_1
      sample mu_j
(5) for each additional xi coordinate:                   (if n_max > n_logup, or n_logup = 0)
      sample xi_k
```

This is GKR's portion of the reference verifier schedule in `verify_zerocheck_and_logup` →
`verify_gkr` → `verify_gkr_sumcheck`. The reference verifier samples alpha and beta
consecutively; in the circuit, alpha is sampled by GKR while beta is reserved for BC (the gap
at `tidx_if`).

The reference verifier steps are:

1. **PoW check:** verify the PoW sample (derived from observing `logup_pow_witness`) satisfies
   the difficulty condition via `ExpBitsLenBus` (if `pow_bits > 0`).
2. **Sample `alpha_logup`** from the transcript.
3. **(gap for `beta_logup`, sampled by BC)**
4. **`verify_gkr`:** for each layer `j = 0..K-1`:
   - Root layer: check `p_cross = 0`, `q_cross = q0_claim`
   - Non-root layers: verify sumcheck (`verify_gkr_sumcheck`), then linear interpolation
     with `mu_j` to get the next layer's claim
   - Each sumcheck round: cubic interpolation at sampled challenge, eq update, final
     consistency check
5. **Xi extension:** sample additional challenges until `xi` has `n_global + l_skip` entries.

## Module-level argument

Each reference verifier check is mapped to the responsible AIR(s):

- **PoW check** on the PoW sample (derived from `logup_pow_witness`): GkrInputAir.
- **Root layer zero-check** (`p_cross = 0`, `q_cross = q0_claim`): GkrLayerAir
  (root-layer boundary constraints).
- **Per-layer sumcheck** (cubic interpolation at sampled challenge, eq update):
  GkrLayerSumcheckAir. **Final consistency check** (verifying the sumcheck output
  equals `(p_cross + lambda * q_cross) * eq_at_r_prime`): GkrLayerAir (via
  `GkrSumcheckOutputBus` receive constraint).
- **Inter-layer claim** (linear interpolation with `mu_j`): GkrLayerAir.
- **Final output** (`gkr_claim = [p_claim, q_claim - alpha_logup]`): GkrInputAir
  (receives last layer's claim via GkrLayerOutputBus, assembles and sends on
  `BatchConstraintModuleBus`).
- **Xi vector assembly** from the last layer's challenges plus any extension:
  GkrLayerAir (sends `mu_{K-1}` on `XiRandomnessBus`), GkrLayerSumcheckAir (sends
  sumcheck challenges), GkrXiSamplerAir (samples and sends any additional coordinates).

**Internal buses:** `GkrLayerInputBus` (GkrInputAir → GkrLayerAir),
`GkrLayerOutputBus` (GkrLayerAir → GkrInputAir),
`GkrSumcheckInputBus` / `GkrSumcheckOutputBus` (GkrLayerAir ↔ GkrLayerSumcheckAir),
`GkrSumcheckChallengeBus` (GkrLayerAir → GkrLayerSumcheckAir, inter-layer challenge passing),
`GkrXiSamplerBus` (GkrInputAir ↔ GkrXiSamplerAir).
Defined in `gkr/bus.rs`.

**Lookup dependencies:** `ExpBitsLenBus` (for PoW check).
