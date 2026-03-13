# Bus Inventory: Recursion Circuit

This document catalogs the buses used in the recursion circuit's 39 AIRs. Buses are the typed message-passing channels over which AIRs communicate. All bus definitions live in `openvm/crates/recursion/src/bus.rs` (global buses) and in module-local `bus.rs` files within each module directory.

## Bus Types

The recursion circuit uses two fundamental bus types:

### Permutation Buses

A **permutation bus** enforces multiset equality between the *sent* messages and the *received* messages. This is the primary mechanism for passing data between AIRs. Note that in some cases (e.g., PublicValuesBus in continuations mode), a message may be sent more than once, so long as the overall multiset of sends matches the multiset of receives.

### Lookup Buses

A **lookup bus** enforces that every looked-up key exists in a provider's table. The table provider adds keys with an integer multiplicity count (the number of lookups it expects to serve), and consumers look up keys with boolean multiplicity. This is used when multiple AIRs need to read the same shared data -- the provider publishes a table and any number of consumers can query it.

### Per-Proof Semantics

Most buses are **per-proof**: the recursion circuit verifies multiple child proofs in a single execution, and each proof has an independent set of bus messages. Per-proof buses automatically prepend a `proof_idx` field to every message, ensuring that messages from different proofs never interact. A bus labeled "global" omits this prefix and its messages are shared across all proofs.

### Notation

Throughout this document:
- `F` denotes a base field element (BabyBear).
- `[F; D_EF]` denotes an extension field element represented as `D_EF = 4` base field elements.
- `[F; DIGEST_SIZE]` denotes a Poseidon2 hash digest (`DIGEST_SIZE = 8`).
- `[F; POSEIDON2_WIDTH]` denotes a full Poseidon2 state (`POSEIDON2_WIDTH = 16`).

---

## 1. Module Control Buses

These buses carry handoff signals between the major verification modules, sequencing the pipeline: ProofShape -> GKR -> BatchConstraint -> Stacking -> WHIR.

### 1.1 TranscriptBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `bus.rs` |
| **Message** | `{tidx: F, value: F, is_sample: F}` |

Implements the Fiat-Shamir transcript as a bus. TranscriptAir is the sole sender; it replays the complete transcript log, emitting one message per field element observed into or sampled from the Poseidon2 duplex sponge. All other AIRs are receivers -- whenever an AIR observes a value (commitment, public value, etc.) or samples a challenge, it receives the corresponding message from this bus.

**Send set:** For each proof, exactly the Fiat-Shamir transcript log: one message per field element, with `is_sample = 0` for observes and `is_sample = 1` for samples, ordered by `tidx`.

**Producers:** TranscriptAir (send).
**Consumers:** All AIRs that observe or sample transcript values (receive).

**Invariants:**
- `tidx` values within a proof are contiguous starting from 0.
- `is_sample` is boolean (0 or 1).
- Helper methods `observe_ext` and `sample_ext` emit `D_EF` consecutive messages (one per base-field coefficient) with consecutive `tidx` values.
- `observe_commit` emits `DIGEST_SIZE` consecutive messages.

---

### 1.2 GkrModuleBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `bus.rs` |
| **Message** | `{tidx: F, n_logup: F, n_max: F, is_n_max_greater: F}` |

Handoff from the ProofShape module to the GKR module. Carries the GKR parameters determined by the proof shape so the GKR module knows the dimensions of the LogUp argument.

**Send set:** Exactly one message per proof containing the GKR parameters.

**Producers:** ProofShapeAir (send).
**Consumers:** GkrInputAir (receive).

**Invariants:**
- `n_logup` is the LogUp hypercube dimension.
- `n_max` is the maximum hypercube dimension across all AIRs.
- `is_n_max_greater` is boolean: 1 if `n_max > n_logup`, 0 otherwise.
- Exactly one message per proof.

---

### 1.3 BatchConstraintModuleBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `bus.rs` |
| **Message** | `{tidx: F, gkr_input_layer_claim: [[F; D_EF]; 2]}` |

Handoff from the GKR module to the BatchConstraint module. Carries the GKR output -- the input-layer claims (numerator and denominator) -- so the batch constraint sumcheck can verify them.

**Send set:** Exactly one message per proof, containing the two extension-field claims from the GKR input layer.

**Producers:** GkrInputAir (send).
**Consumers:** FractionsFolderAir (receive).

**Invariants:**
- `gkr_input_layer_claim[0]` is the numerator claim; `gkr_input_layer_claim[1]` is the denominator claim.
- Exactly one message per proof.

---

### 1.4 StackingModuleBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `bus.rs` |
| **Message** | `{tidx: F}` |

Handoff from the BatchConstraint sumcheck to the Stacking module. Signals that the batch constraint sumcheck is complete and provides the transcript index for continuation.

**Send set:** One message per proof.

**Producers:** UnivariateSumcheckAir or MultilinearSumcheckAir (send).
**Consumers:** OpeningClaimsAir, MultilinearSumcheckAir (receive).

**Invariants:**
- Exactly one message per proof.
- `tidx` is the transcript index after the batch constraint sumcheck completes.

---

### 1.5 WhirModuleBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `bus.rs` |
| **Message** | `{tidx: F, claim: [F; 4]}` |

Handoff from the Stacking module to the WHIR module. Carries the batched opening claim after mu-folding, which WHIR will verify via its FRI-like protocol.

**Send set:** One message per proof.

**Producers:** StackingClaimsAir (send).
**Consumers:** WhirRoundAir (receive).

**Invariants:**
- `claim` is the batched opening claim (extension field element) after stacking and mu-batching.
- `tidx` is the transcript index after the mu batching randomness is sampled.

---

### 1.6 WhirMuBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `bus.rs` |
| **Message** | `{mu: [F; D_EF]}` |

Passes the mu batching challenge from the Stacking module to the WHIR module. The mu challenge is used to combine multiple stacking claims into a single opening claim.

**Send set:** One message per proof.

**Producers:** StackingClaimsAir (send).
**Consumers:** InitialOpenedValuesAir (receive).

**Invariants:**
- `mu` is the extension field batching challenge sampled from the transcript.

---

### 1.7 MerkleVerifyBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `bus.rs` |
| **Message** | `{merkle_idx_bit_src: F, current_idx_bit_src: F, total_depth: F, height: F, is_leaf: F, leaf_sub_idx: F, value: [F; DIGEST_SIZE], commit_major: F, commit_minor: F}` |

Permutation bus for Merkle tree verification. MerkleVerifyAir both sends and receives on this bus: at each level of a Merkle proof, it receives left and right children and sends the computed parent hash. The root hash at the top of each proof is verified against the CommitmentsBus. MerkleVerifyAir uses single-row constraints only (no next-row access).

**Send set:** For each Merkle proof, one message per node in the authentication path, from leaves to root.

**Producers:** MerkleVerifyAir (send parent hashes), InitialOpenedValuesAir and NonInitialOpenedValuesAir (send leaf hashes).
**Consumers:** MerkleVerifyAir (receive child hashes).

**Invariants:**
- `merkle_idx_bit_src` is the Merkle index used for bit-source shifting (determines left/right placement via parity at each level).
- `current_idx_bit_src` is `merkle_idx_bit_src >> max(0, total_depth - k)`, the index after right-shifting by the leaf tree depth.
- `total_depth` is the full depth of the Merkle tree including the leaf-hashing part.
- `height` ranges from `[0, k)` for leaf hashing and `[k, total_depth)` for the authentication path.
- `is_leaf` is boolean: 1 for rows in the leaf-combining phase, 0 for authentication path rows.
- `leaf_sub_idx` ranges from `0` to `2^k - 1` for leaves, `0` to `2^{k-1} - 1` for intermediate values, and `0` for the authentication path.
- `commit_major` and `commit_minor` identify the commitment this Merkle proof authenticates against.

---

## 2. Transcript and Cryptographic Buses

These buses provide shared cryptographic primitive lookup tables.

### 2.1 Poseidon2PermuteBus

| Property | Value |
|---|---|
| **Type** | Lookup, global |
| **Source** | `bus.rs` |
| **Message** | `{input: [F; 16], output: [F; 16]}` |

Lookup table of Poseidon2 permutation evaluations. Any AIR that needs to evaluate the Poseidon2 permutation looks up the (input, output) pair in this table rather than re-computing it.

**Table:** All `(input, Poseidon2(input))` pairs needed across all proofs.

**Provider:** Poseidon2Air.
**Consumers:** TranscriptAir (for sponge operations), InitialOpenedValuesAir (for leaf hashing of initial opened rows).

**Invariants:**
- `output = Poseidon2_permute(input)` for every entry in the table.
- The table size equals the total number of Poseidon2 permutation calls across all proofs.

---

### 2.2 Poseidon2CompressBus

| Property | Value |
|---|---|
| **Type** | Lookup, global |
| **Source** | `bus.rs` |
| **Message** | `{input: [F; 16], output: [F; 8]}` |

Lookup table of Poseidon2 compression evaluations. Used for Merkle tree hashing where two 8-element digests are compressed into one.

**Table:** All `(input, first_8(Poseidon2(input)))` pairs needed across all proofs.

**Provider:** Poseidon2Air.
**Consumers:** MerkleVerifyAir (for authentication path compression), NonInitialOpenedValuesAir (for codeword leaf hashing).

**Invariants:**
- `output = Poseidon2_permute(input)[0..8]` for every entry.

---

## 3. Data / Shape Buses

These buses distribute proof metadata -- shapes, dimensions, commitments, and public values -- from the ProofShape module to downstream consumers.

### 3.1 AirShapeBus

| Property | Value |
|---|---|
| **Type** | Lookup, per-proof |
| **Source** | `bus.rs` |
| **Message** | `{sort_idx: F, property_idx: F, value: F}` |

Lookup table of AIR shape properties. For each AIR present in the child proof (indexed by `sort_idx`), this table stores properties such as the AIR identifier, number of interactions, and whether the AIR needs rotation.

**Table:** For each present AIR, one entry per property:
- `property_idx = 0` (AirId): the AIR's identifier.
- `property_idx = 1` (NumInteractions): the number of bus interactions for this AIR.
- `property_idx = 2` (NeedRot): 1 if the AIR needs rotated trace access, 0 otherwise.

**Provider:** ProofShapeAir.
**Consumers:** SymbolicExpressionAir, InteractionsFoldingAir, OpeningClaimsAir.

**Invariants:**
- `sort_idx` values are contiguous within a proof.
- `property_idx` is in `{0, 1, 2}`.

---

### 3.2 HyperdimBus

| Property | Value |
|---|---|
| **Type** | Lookup, per-proof |
| **Source** | `bus.rs` |
| **Message** | `{sort_idx: F, n_abs: F, n_sign_bit: F}` |

Lookup table of hypercube dimensions per AIR. The hypercube dimension `n` for an AIR is `log_height - l_skip`, which can be negative if the trace height is smaller than `2^l_skip`.

**Table:** For each present AIR, one entry with `n_abs = |log_height - l_skip|` and `n_sign_bit = 1` if `log_height < l_skip` (i.e., `n < 0`), `0` otherwise.

**Provider:** ProofShapeAir.
**Consumers:** SymbolicExpressionAir, ExpressionClaimAir.

**Invariants:**
- `n_sign_bit` is boolean.
- `n_abs = |log_height - l_skip|`.

---

### 3.3 LiftedHeightsBus

| Property | Value |
|---|---|
| **Type** | Lookup, per-proof |
| **Source** | `bus.rs` |
| **Message** | `{sort_idx: F, part_idx: F, commit_idx: F, hypercube_dim: F, lifted_height: F, log_lifted_height: F}` |

Lookup table of lifted trace heights per column partition. Each AIR trace is partitioned into column groups, and the "lifted height" is `max(trace_height, 2^l_skip)` -- the effective height after lifting short traces to the minimum stacking height.

**Table:** For each present AIR and each column partition, one entry with the commit index, hypercube dimension, and lifted height.

**Provider:** ProofShapeAir.
**Consumer:** OpeningClaimsAir.

**Invariants:**
- `lifted_height = 2^log_lifted_height`.
- `log_lifted_height = max(log_height, l_skip)`.

---

### 3.4 CommitmentsBus

| Property | Value |
|---|---|
| **Type** | Lookup, per-proof |
| **Source** | `bus.rs` |
| **Message** | `{major_idx: F, minor_idx: F, commitment: [F; 8]}` |

Lookup table of all Merkle root commitments in a proof. Stacking commitments use `major_idx = 0` with `minor_idx` as the stacking matrix index. WHIR round commitments use `major_idx = whir_round + 1` with `minor_idx = 0`.

**Table:** All commitment roots for both the stacking phase and each WHIR round.

**Providers:** ProofShapeAir (stacking commitments), WhirRoundAir (WHIR round commitments).
**Consumer:** MerkleVerifyAir (verifies that authentication paths lead to the correct root).

**Invariants:**
- `commitment` is a valid Poseidon2 digest (`DIGEST_SIZE = 8` field elements).
- Stacking: `major_idx = 0`, `minor_idx` in `{0, ..., num_stacking_matrices - 1}`.
- WHIR: `major_idx = round + 1`, `minor_idx = 0`.

---

### 3.5 PublicValuesBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `bus.rs` |
| **Message** | `{air_idx: F, pv_idx: F, value: F}` |

Carries public values from each AIR in the child proof to the constraint evaluation module.

**Send set:** One message per public-value field element across all AIRs in the proof.

**Producers:** PublicValuesAir (send).
**Consumers:** SymbolicExpressionAir (receive, to use public values in constraint evaluation).

**Invariants:**
- `air_idx` identifies which child AIR the public value belongs to.
- `pv_idx` is the index within that AIR's public values vector.

---

### 3.6 StackingIndicesBus

| Property | Value |
|---|---|
| **Type** | Lookup, per-proof |
| **Source** | `bus.rs` |
| **Message** | `{commit_idx: F, col_idx: F}` |

Lookup table mapping stacked columns to their commit indices. When traces are stacked into matrices for commitment, this table records which commit index each column belongs to.

**Table:** For each stacked column, its `(commit_idx, col_idx)` pair.

**Provider:** StackingClaimsAir.
**Consumer:** InitialOpenedValuesAir.

**Invariants:**
- `commit_idx` and `col_idx` are non-negative integers.
- Each `(commit_idx, col_idx)` pair appears exactly once per proof in the table.

---

### 3.7 ColumnClaimsBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `bus.rs` |
| **Message** | `{sort_idx: F, part_idx: F, col_idx: F, claim: [F; D_EF], is_rot: F}` |

Carries per-column opening claims from the stacking module to downstream verification. Each message associates a column (identified by sort index, partition, and column index) with its claimed evaluation at the opening point.

**Send set:** One message per column opening claim, including rotation claims where applicable.

**Producers:** OpeningClaimsAir (send).
**Consumers:** SymbolicExpressionAir (receive).

**Invariants:**
- `is_rot` is boolean: 1 if this claim corresponds to a rotated column evaluation, 0 otherwise.
- `claim` is an extension field element.

---

## 4. Randomness / Challenge Buses

These buses distribute random challenges sampled from the Fiat-Shamir transcript to the AIRs that consume them.

### 4.1 XiRandomnessBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `bus.rs` |
| **Message** | `{idx: F, xi: [F; D_EF]}` |

Carries xi challenge vectors used in the GKR protocol. Each xi value is an extension-field random challenge used to combine GKR layer claims.

**Send set:** One `(idx, xi)` pair for each challenge index used in the GKR protocol.

**Producers:** GkrLayerAir, GkrLayerSumcheckAir, GkrXiSamplerAir (send).
**Consumers:** EqNsAir, EqSharpUniAir (receive).

**Invariants:**
- `idx` values are distinct within a proof.
- `xi` values are extension field elements sampled from the transcript.

---

### 4.2 ConstraintSumcheckRandomnessBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `bus.rs` |
| **Message** | `{idx: F, challenge: [F; D_EF]}` |

Carries the sumcheck round challenges `r_0, r_1, ..., r_{n_max}` used in the batch constraint sumcheck protocol.

**Send set:** One message per sumcheck round.

**Producers:** UnivariateSumcheckAir (sends `r_0`, the first-round challenge), MultilinearSumcheckAir (sends `r_1` through `r_{n_max}`).
**Consumers:** SumcheckRoundsAir, EqBaseAir.

**Invariants:**
- `idx` ranges from 0 to `n_max`.
- `challenge` values are sampled from the Fiat-Shamir transcript.

---

### 4.3 WhirOpeningPointBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `bus.rs` |
| **Message** | `{idx: F, value: [F; D_EF]}` |

Carries opening points produced by the stacking sumcheck and eq_base AIRs to the WHIR module. Each point is sent exactly once by the producer and received exactly once by the first node in the corresponding layer of the MLE evaluation tree.

**Send set:** One opening point per sumcheck round.

**Producers:** SumcheckRoundsAir, EqBaseAir (send).
**Consumers:** SumcheckAir (WHIR), FinalPolyMleEvalAir (receive).

**Invariants:**
- Each `idx` appears exactly once as a send and once as a receive.

---

### 4.4 WhirOpeningPointLookupBus

| Property | Value |
|---|---|
| **Type** | Lookup, per-proof |
| **Source** | `bus.rs` |
| **Message** | `{idx: F, value: [F; D_EF]}` |

Lookup version of the opening points bus, used for distributing opening points within the MLE evaluation tree layers. The first node in each layer registers the point (received via WhirOpeningPointBus) as a lookup key, and remaining nodes in the layer look it up. Internal to FinalPolyMleEvalAir.

**Table:** Opening points indexed by layer, one entry per layer.

**Provider:** FinalPolyMleEvalAir (add_key_with_lookups, first node in each layer).
**Consumers:** FinalPolyMleEvalAir (lookup_key, remaining nodes in each layer).

**Invariants:**
- `idx` indexes the opening point dimensions.
- Each `idx` in `0..num_total_sumcheck_rounds` appears exactly once.
- The `value` is the stacking/WHIR sumcheck challenge at that round.

---

### 4.5 EqNegBaseRandBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `bus.rs` |
| **Message** | `{u: [F; D_EF], r_squared: [F; D_EF]}` |

Carries base randomness for the negative-dimension equality polynomial computation. Passes the sampled values `u_0` and `r_0^2` needed to evaluate `eq_n` for negative hypercube dimensions.

**Send set:** One message per proof.

**Producers:** EqBaseAir (stacking module) (send).
**Consumers:** EqNegAir (batch constraint module) (receive).

**Invariants:**
- `u` is the sampled extension field value `u_0`.
- `r_squared` is `r_0^2` in the extension field.

---

## 5. Computation Buses

These buses support shared computations -- equality polynomial evaluations, exponentiation, and selection functions.

### 5.1 ExpBitsLenBus

| Property | Value |
|---|---|
| **Type** | Lookup, global |
| **Source** | `primitives/bus.rs` |
| **Message** | `{base: F, bit_src: F, num_bits: F, result: F}` |

Lookup table for partial exponentiation: `result = base^(bit_src & ((1 << num_bits) - 1))`. This computes a power of `base` determined by the lowest `num_bits` bits of `bit_src`.

**Table:** `{(base, bit_src, num_bits, result) : result = base^(bit_src & ((1 << num_bits) - 1))}`. The table computes modular exponentiation of `base` raised to the low `num_bits` bits of `bit_src`.

**Provider:** ExpBitsLenAir.
**Consumers:** GkrInputAir, StackingClaimsAir, WhirRoundAir, SumcheckAir (WHIR), WhirQueryAir.

**Invariants:**
- `result = base^(bit_src & ((1 << num_bits) - 1))`.
- `num_bits` is a small positive integer.

---

### 5.2 RangeCheckerBus

| Property | Value |
|---|---|
| **Type** | Lookup, global |
| **Source** | `primitives/bus.rs` |
| **Message** | `{value: F, max_bits: F}` |

Lookup table for range checks. Verifies that a value fits within the specified number of bits.

**Table:** The table consists of `{(i, NUM_BITS) : 0 <= i <= 255}` where `NUM_BITS = 8`. The `max_bits` field is always 8; every key has `max_bits = NUM_BITS`.

**Provider:** RangeCheckerAir (provides the 256-row table of values 0..255).
**Consumers:** ProofShapeAir, PowerCheckerAir.

---

### 5.3 PowerCheckerBus

| Property | Value |
|---|---|
| **Type** | Lookup, global |
| **Source** | `primitives/bus.rs` |
| **Message** | `{log: F, exp: F}` |

Lookup table of powers of two: `exp = 2^log`.

**Table:** `{(i, 2^i) : i in 0..POW_CHECKER_HEIGHT}` (typically `i in 0..32`).

**Provider:** PowerCheckerAir.
**Consumers:** ProofShapeAir, ExpressionClaimAir.

**Invariants:**
- `exp = 2^log` for every entry.

---

### 5.4 SelUniBus

| Property | Value |
|---|---|
| **Type** | Lookup, per-proof |
| **Source** | `bus.rs` |
| **Message** | `{n: F, is_first: F, value: [F; D_EF]}` |

Lookup table of univariate selector polynomial evaluations. Used in the batch constraint module for evaluating the univariate selector at sumcheck points.

**Table:** `{(n, is_first, v) : v = sel_uni(n, is_first, r, omega)}` where `sel_uni(n, 0, r, omega) = prod_{i=0}^{l_skip+n-1}(r^{2^i})` (product of r-powers) and `sel_uni(n, 1, r, omega) = prod(1-r^{2^i})` (product of one-minus-r-powers), derived from the univariate selector polynomial for the first/last row at hypercube dimension `n`.

**Provider:** EqNegAir.
**Consumers:** SymbolicExpressionAir.

**Invariants:**
- `is_first` is boolean.
- `value` is an extension field element.

---

### 5.5 SelHypercubeBus

| Property | Value |
|---|---|
| **Type** | Lookup, per-proof |
| **Source** | `bus.rs` |
| **Message** | `{n: F, is_first: F, value: [F; D_EF]}` |

Lookup table of hypercube selector polynomial evaluations. Similar to SelUniBus but for the hypercube selector function.

**Table:** Selector values indexed by hypercube dimension and position.

**Provider:** EqNsAir.
**Consumers:** SymbolicExpressionAir.

**Invariants:**
- `is_first` is boolean (0 or 1).
- `n` is the hypercube dimension (non-negative integer).
- The lookup value is `sel_hypercube(n, is_first, r_prefix) = product(r_i)` when `is_first=0`, `product(1-r_i)` when `is_first=1`, evaluated over the prefix of sumcheck randomness.

---

### 5.6 EqNegResultBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `bus.rs` |
| **Message** | `{n: F, eq: [F; D_EF], k_rot: [F; D_EF]}` |

Carries results of the negative-dimension equality polynomial computation. For AIRs with `log_height < l_skip` (negative hypercube dimension `n`), this bus passes the pre-computed `eq_n` and rotation kernel `k_rot_n` values.

**Send set:** One message per distinct negative hypercube dimension.

**Producers:** EqNegAir (send).
**Consumers:** EqBaseAir (stacking) (receive).

**Invariants:**
- `n` is the (negative) hypercube dimension, i.e., `n < 0`. The sender computes `n = -neg_hypercube` where `neg_hypercube > 0`.
- `eq` = `2^{l_skip + n} * eq_n(u_0, r_0)`.
- `k_rot` = `2^{l_skip + n} * k_rot_n(u_0, r_0)`.

---

### 5.7 ExpressionClaimNMaxBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `bus.rs` |
| **Message** | `{n_max: F}` |

Passes the maximum hypercube dimension `n_max` from the proof shape to the expression claim computation. This single value determines the sumcheck degree.

**Send set:** One message per proof.

**Producers:** ProofShapeAir (send).
**Consumers:** ExpressionClaimAir (receive).

---

### 5.8 FractionFolderInputBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `bus.rs` |
| **Message** | `{num_present_airs: F}` |

Passes the count of present AIRs from the proof shape to the fractions folder, which needs to know how many constraint and interaction fractions to combine.

**Send set:** One message per proof.

**Producers:** ProofShapeAir (send).
**Consumers:** FractionsFolderAir (receive).

---

### 5.9 NLiftBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `bus.rs` |
| **Message** | `{air_idx: F, n_lift: F}` |

Passes the lift amount `n_lift` for each AIR. When an AIR's trace height is less than `2^l_skip`, the trace is "lifted" (zero-padded) to the minimum stacking height. `n_lift = max(0, l_skip - log_height)`.

**Send set:** One message per AIR in the proof.

**Producers:** ProofShapeAir (send).
**Consumers:** ConstraintsFoldingAir (receive).

---

### 5.10 AirPresenceBus

| Property | Value |
|---|---|
| **Type** | Lookup, per-proof |
| **Source** | `bus.rs` |
| **Message** | `{air_idx: F, is_present: F}` |

Lookup table of AIR presence flags. For each possible AIR in the child proof, records whether it is present. Used by SymbolicExpressionAir to determine which AIRs to evaluate.

**Table:** For each AIR index, its presence flag (0 or 1).

**Provider:** ProofShapeAir.
**Consumers:** SymbolicExpressionAir.

**Invariants:**
- `is_present` is boolean.
- One entry per possible AIR index.

---

### 5.11 Eq3bShapeBus

| Property | Value |
|---|---|
| **Type** | Lookup, per-proof |
| **Source** | `bus.rs` |
| **Message** | `{sort_idx: F, n_lift: F, n_logup: F}` |

Passes the shape parameters `n_lift` and `n_logup` for each AIR to the Eq3bAir, which needs these to determine the range of hypercube dimensions over which to compute equality polynomial evaluations.

**Send set:** One message per present AIR.

**Producers:** ProofShapeAir (send).
**Consumers:** Eq3bAir (receive).

---

### 5.12 EqNsNLogupMaxBus

| Property | Value |
|---|---|
| **Type** | Lookup, per-proof |
| **Source** | `bus.rs` |
| **Message** | `{n_logup: F, n_max: F}` |

Passes the global LogUp dimension `n_logup` and the maximum hypercube dimension `n_max` to EqNsAir, so it knows the range of dimensions for multivariate equality polynomial computation.

**Send set:** One message per proof.

**Producers:** ProofShapeAir (send).
**Consumers:** EqNsAir (receive).

---

### 5.13 ConstraintsFoldingInputBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `bus.rs` |
| **Message** | `{tidx: F}` |

Passes the transcript index for the lambda challenge to ConstraintsFoldingAir. This allows the folding AIR to locate where in the transcript the lambda challenge was sampled.

**Send set:** One message per proof.

**Producers:** ProofShapeAir (send).
**Consumers:** ConstraintsFoldingAir (receive).

---

### 5.14 InteractionsFoldingInputBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `bus.rs` |
| **Message** | `{tidx: F}` |

Passes the transcript index for the beta challenge to InteractionsFoldingAir. This allows the folding AIR to locate where in the transcript the beta challenge was sampled.

**Send set:** One message per proof.

**Producers:** ProofShapeAir (send).
**Consumers:** InteractionsFoldingAir (receive).

---

### 5.15 FinalTranscriptStateBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `bus.rs` |
| **Message** | `{state: [F; POSEIDON2_WIDTH]}` |

Carries the final Poseidon2 sponge state from the transcript to the continuations circuit. Only active when continuations are enabled. Allows the next continuation segment to resume the transcript from the correct sponge state.

**Send set:** One message per proof (on the last transcript row).

**Producers:** TranscriptAir (send, continuations only).
**Consumers:** Continuations circuit (receive).

---

### 5.16 PreHashBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `bus.rs` |
| **Message** | `{vk_pre_hash: [F; DIGEST_SIZE]}` |

Carries the child VK pre-hash for incorporation into the inner verifier's public values. Only active when continuations are enabled.

**Send set:** One message per proof.

**Producers:** ProofShapeAir (send, continuations only).
**Consumers:** Continuations circuit (receive).

---

## 6. Internal Module Buses

These buses are used only within a single module and are not part of the global `BusInventory`. They are allocated locally by each module from the shared `BusIndexManager`.

---

### 6.1 Proof Shape Module Internal Buses

#### 6.1.1 ProofShapePermutationBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `proof_shape/bus.rs` |
| **Message** | `{idx: F}` |

Internal sequencing bus within the ProofShapeAir. Used to enforce ordering of shape entries by sending and receiving sequential index values.

**Send set:** One message per row index.

**Producers:** ProofShapeAir (internal send/receive).

**Invariants:**
- Exactly one message per proof; the send and receive balance ensures exactly one summary row per proof.

---

#### 6.1.2 StartingTidxBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `proof_shape/bus.rs` |
| **Message** | `{air_idx: F, tidx: F}` |

Internal bus within ProofShapeAir. Used to pass starting transcript indices between different phases of the proof shape computation. Both the send and receive are within ProofShapeAir itself.

**Send set:** One message per AIR.

**Producers:** ProofShapeAir (send).
**Consumers:** ProofShapeAir (receive).

---

#### 6.1.3 NumPublicValuesBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `proof_shape/bus.rs` |
| **Message** | `{air_idx: F, tidx: F, num_pvs: F}` |

Carries the number of public values for each AIR along with the transcript index at which public value observation begins.

**Send set:** One message per AIR.

**Producers:** ProofShapeAir (send).
**Consumers:** PublicValuesAir (receive).

---

### 6.2 GKR Module Internal Buses

#### 6.2.1 GkrXiSamplerBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `gkr/bus.rs` |
| **Message** | `{idx: F, tidx: F}` |

Coordinates xi challenge sampling between the GKR layer processing and the xi sampler. Passes the challenge index and corresponding transcript index.

**Send set:** One message per xi challenge.

**Producers:** GkrLayerAir (send).
**Consumers:** GkrXiSamplerAir (receive).

---

#### 6.2.2 GkrLayerInputBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `gkr/bus.rs` |
| **Message** | `{tidx: F, q0_claim: [F; D_EF]}` |

Carries the initial claim from GkrInputAir to GkrLayerAir, providing the starting point for the GKR layer-by-layer verification.

**Send set:** One message per proof.

**Producers:** GkrInputAir (send).
**Consumers:** GkrLayerAir (receive).

**Invariants:**
- `q0_claim` is the initial GKR claim (layer 0) as an extension field element.

---

#### 6.2.3 GkrLayerOutputBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `gkr/bus.rs` |
| **Message** | `{tidx: F, layer_idx_end: F, input_layer_claim: [[F; D_EF]; 2]}` |

Carries the final GKR layer output back from GkrLayerAir to GkrInputAir. Contains the input-layer claims (numerator and denominator) that will be forwarded to the batch constraint module.

**Send set:** One message per proof.

**Producers:** GkrLayerAir (send).
**Consumers:** GkrInputAir (receive).

**Invariants:**
- `input_layer_claim[0]` is the numerator claim; `input_layer_claim[1]` is the denominator claim.
- `layer_idx_end` is the total number of GKR layers processed.

---

#### 6.2.4 GkrSumcheckInputBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `gkr/bus.rs` |
| **Message** | `{layer_idx: F, is_last_layer: F, tidx: F, claim: [F; D_EF]}` |

Sends the combined claim for a GKR layer to the GkrLayerSumcheckAir for sumcheck verification.

**Send set:** One message per GKR layer.

**Producers:** GkrLayerAir (send).
**Consumers:** GkrLayerSumcheckAir (receive).

**Invariants:**
- `is_last_layer` is boolean.
- `claim` is the combined claim to be verified by sumcheck.

---

#### 6.2.5 GkrSumcheckOutputBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `gkr/bus.rs` |
| **Message** | `{layer_idx: F, tidx: F, claim_out: [F; D_EF], eq_at_r_prime: [F; D_EF]}` |

Returns the sumcheck result from GkrLayerSumcheckAir back to GkrLayerAir: the reduced claim and the equality polynomial evaluation at `r'`.

**Send set:** One message per GKR layer.

**Producers:** GkrLayerSumcheckAir (send).
**Consumers:** GkrLayerAir (receive).

**Invariants:**
- `claim_out` is the new reduced claim after sumcheck.
- `eq_at_r_prime` is `eq(r, r')` evaluated at the sumcheck output point.

---

#### 6.2.6 GkrSumcheckChallengeBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `gkr/bus.rs` |
| **Message** | `{layer_idx: F, sumcheck_round: F, challenge: [F; D_EF]}` |

Passes challenges between consecutive GKR sumcheck sub-rounds within the same layer.

**Send set:** One message per sumcheck round per GKR layer.

**Producers:** GkrLayerSumcheckAir (send, from one round to the next).
**Consumers:** GkrLayerSumcheckAir (receive, at the start of the next round).

**Invariants:**
- `layer_idx` identifies the GKR layer.
- `sumcheck_round` identifies the round within that layer's sumcheck.

---

### 6.3 Batch Constraint Module Internal Buses

#### 6.3.1 BatchConstraintConductorBus

| Property | Value |
|---|---|
| **Type** | Lookup, per-proof |
| **Source** | `batch_constraint/bus.rs` |
| **Message** | `{msg_type: F, idx: F, value: [F; D_EF]}` |

Central distribution bus for the batch constraint module. Distributes three types of values (encoded by `msg_type`) to sub-AIRs:
- `msg_type = 0` (R): Sumcheck round challenges `r_i`.
- `msg_type = 1` (Xi): GKR xi challenges.
- `msg_type = 2` (Mu): Batching randomness mu.

**Table:** All `(msg_type, idx, value)` triples needed by batch constraint sub-AIRs.

**Provider:** FractionsFolderAir or conductor logic.
**Consumers:** Various batch constraint sub-AIRs that need access to challenges.

---

#### 6.3.2 SumcheckClaimBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `batch_constraint/bus.rs` |
| **Message** | `{round: F, value: [F; D_EF]}` |

Carries sumcheck claims between rounds in the batch constraint sumcheck. Each message contains the claim value for a particular round, allowing the univariate and multilinear sumcheck AIRs to chain their outputs.

**Send set:** One message per sumcheck round.

**Producers:** UnivariateSumcheckAir, MultilinearSumcheckAir (send).
**Consumers:** UnivariateSumcheckAir, MultilinearSumcheckAir (receive, for the next round).

---

#### 6.3.3 SymbolicExpressionBus

| Property | Value |
|---|---|
| **Type** | Lookup, per-proof |
| **Source** | `batch_constraint/bus.rs` |
| **Message** | `{air_idx: F, node_idx: F, value: [F; D_EF]}` |

Lookup table of symbolic expression DAG node evaluations. The SymbolicExpressionAir evaluates the constraint expression DAG for each child AIR, and this bus allows sharing of intermediate node values.

**Table:** For each AIR and each DAG node, the evaluated extension field value.

**Provider:** SymbolicExpressionAir (add_key_with_lookups).
**Consumers:** SymbolicExpressionAir (self-lookups for DAG node sharing).

**Invariants:**
- `node_idx` identifies a node in the symbolic expression DAG.
- `value` is the evaluated result at that node for the given AIR.

---

#### 6.3.4 ExpressionClaimBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `batch_constraint/bus.rs` |
| **Message** | `{is_interaction: F, idx: F, value: [F; D_EF]}` |

Carries evaluated constraint and interaction expression claims. Each message contains either a constraint expression value or an interaction expression value (distinguished by `is_interaction`).

**Send set:** One message per constraint expression and one per interaction expression for each AIR.

**Producers:** InteractionsFoldingAir (send), ConstraintsFoldingAir (send).
**Consumers:** ExpressionClaimAir (receive).

**Invariants:**
- `is_interaction` is boolean: 1 for interaction expressions, 0 for constraint expressions.

---

#### 6.3.5 InteractionsFoldingBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `batch_constraint/bus.rs` |
| **Message** | `{air_idx: F, interaction_idx: F, is_mult: F, idx_in_message: F, value: [F; D_EF]}` |

Carries folded interaction field values. Each bus interaction has a message (column values) and a multiplicity; this bus carries the individual field elements of those messages and multiplicities after folding.

**Send set:** One message per field element in each interaction's message and multiplicity, for each AIR.

**Producers:** SymbolicExpressionAir (send).
**Consumers:** InteractionsFoldingAir (receive).

**Invariants:**
- `is_mult` is boolean: 1 for multiplicity values, 0 for message values.
- `idx_in_message` is the index within the interaction's message vector.

---

#### 6.3.6 ConstraintsFoldingBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `batch_constraint/bus.rs` |
| **Message** | `{air_idx: F, constraint_idx: F, value: [F; D_EF]}` |

Carries folded constraint values from the SymbolicExpressionAir to the ConstraintsFoldingAir. Each constraint's evaluated value is sent as a separate message.

**Send set:** One message per constraint per AIR.

**Producers:** SymbolicExpressionAir (send).
**Consumers:** ConstraintsFoldingAir (receive).

---

#### 6.3.7 Eq3bBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `batch_constraint/bus.rs` |
| **Message** | `{sort_idx: F, interaction_idx: F, eq_3b: [F; D_EF]}` |

Carries the "3b" equality polynomial evaluation for each interaction. This is a partial equality polynomial evaluation used in the GKR-to-constraint-sumcheck reduction.

**Send set:** One message per interaction per AIR.

**Producers:** Eq3bAir (send).
**Consumers:** InteractionsFoldingAir (receive).

**Invariants:**
- `eq_3b` is an extension field value representing a partial eq polynomial product.

---

#### 6.3.8 EqSharpUniBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `batch_constraint/bus.rs` |
| **Message** | `{xi_idx: F, iter_idx: F, product: [F; D_EF]}` |

Carries partial products of the "sharp univariate" equality polynomial. These are intermediate accumulations used when computing `eq_sharp_uni(xi, r)`.

**Send set:** One message per `(xi_idx, iter_idx)` pair.

**Producers:** EqSharpUniAir (send).
**Consumers:** EqNsAir (receive).

**Invariants:**
- The two products (`prod_1` and `prod_2`) represent partial evaluation products from the sharp univariate eq polynomial butterfly expansion.

---

#### 6.3.9 EqZeroNBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `batch_constraint/bus.rs` |
| **Message** | `{is_sharp: F, value: [F; D_EF]}` |

Carries the base case (`n = 0`) of the equality polynomial computation, for both the regular and "sharp" variants.

**Send set:** One message per variant (regular and sharp).

**Producers:** EqUniAir, EqSharpUniReceiverAir (send).
**Consumers:** EqNsAir (receive).

**Invariants:**
- `is_sharp` is boolean: 1 for the sharp variant, 0 for regular.

---

#### 6.3.10 EqNOuterBus

| Property | Value |
|---|---|
| **Type** | Lookup, per-proof |
| **Source** | `batch_constraint/bus.rs` |
| **Message** | `{is_sharp: F, n: F, value: [F; D_EF]}` |

Lookup table of equality polynomial values `eq_n(xi, r)` indexed by dimension `n` and variant (regular or sharp). Allows sub-AIRs to look up pre-computed eq values by dimension.

**Table:** For each dimension `n` and each variant, the evaluated eq polynomial value.

**Provider:** EqNsAir.
**Consumers:** SymbolicExpressionAir, InteractionsFoldingAir.

---

#### 6.3.11 EqNegInternalBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `batch_constraint/bus.rs` |
| **Message** | `{neg_n: F, u: [F; D_EF], r: [F; D_EF], r_omega: [F; D_EF]}` |

Internal bus for the negative-dimension equality polynomial computation. Carries intermediate values between recursive steps of `eq_n` for `n < 0`.

**Send set:** One message per recursive step in the negative-dimension computation.

**Producers:** EqNegAir (send, from one step).
**Consumers:** EqNegAir (receive, at the next step).

**Invariants:**
- `neg_n` is the negated hypercube dimension (a positive value representing `|n|`).
- `u` is the sampled value `u_0`.
- `r` is `r_0^{2^{1 - n}}`.
- `r_omega` is `(r_0 * omega)^{2^{1 - n}}` where `omega` is the degree-`D^2` generator.

---

#### 6.3.12 UnivariateSumcheckInputBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `batch_constraint/bus.rs` |
| **Message** | `{tidx: F}` |

Passes the transcript index to the univariate sumcheck AIR, signaling where in the transcript the univariate sumcheck round begins.

**Send set:** One message per proof.

**Producers:** FractionsFolderAir (send).
**Consumers:** UnivariateSumcheckAir (receive).

---

### 6.4 Stacking Module Internal Buses

#### 6.4.1 StackingModuleTidxBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `stacking/bus.rs` |
| **Message** | `{module_idx: F, tidx: F}` |

Internal transcript index handoff within the stacking module. Passes transcript positions between stacking sub-modules.

**Send set:** One message per sub-module transition.

**Producers:** OpeningClaimsAir, UnivariateRoundAir, SumcheckRoundsAir (send).
**Consumers:** StackingClaimsAir, UnivariateRoundAir, SumcheckRoundsAir (receive).

**Invariants:**
- `module_idx` identifies the stacking sub-module (0=OpeningClaims, 1=UnivariateRound, 2=SumcheckRounds).
- `tidx` is the transcript index at the boundary between sub-modules.

---

#### 6.4.2 ClaimCoefficientsBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `stacking/bus.rs` |
| **Message** | `{commit_idx: F, stacked_col_idx: F, coefficient: [F; D_EF]}` |

Carries the batching coefficients for combining column claims within each stacking commitment. Each column in a stacking matrix gets a coefficient derived from the batching challenge.

**Send set:** One message per stacked column.

**Producers:** OpeningClaimsAir (send).
**Consumers:** StackingClaimsAir (receive).

**Invariants:**
- One message per stacked column.
- `commit_idx` and `stacked_col_idx` uniquely identify each column.

---

#### 6.4.3 SumcheckClaimsBus (Stacking)

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `stacking/bus.rs` |
| **Message** | `{module_idx: F, value: [F; D_EF]}` |

Carries sumcheck claims between stacking sub-modules. The `module_idx` distinguishes claims from different sub-module stages.

**Send set:** One message per sub-module.

**Producers:** SumcheckRoundsAir, UnivariateRoundAir, OpeningClaimsAir (send).
**Consumers:** SumcheckRoundsAir, UnivariateRoundAir, StackingClaimsAir (receive).

**Invariants:**
- `module_idx` indexes the stacking sumcheck stages.
- Exactly one message per proof per `module_idx` boundary.

---

#### 6.4.4 EqRandValuesLookupBus

| Property | Value |
|---|---|
| **Type** | Lookup, per-proof |
| **Source** | `stacking/bus.rs` |
| **Message** | `{idx: F, u: [F; D_EF]}` |

Lookup table of the random values `u_i` used in the stacking equality polynomial. These are the evaluation point coordinates for `eq(u, .)`.

**Table:** For each index `i`, the corresponding `u_i` value.

**Providers:** SumcheckRoundsAir, UnivariateRoundAir (add_key_with_lookups).
**Consumers:** EqBitsAir, EqBaseAir (lookup_key).

---

#### 6.4.5 EqBaseBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `stacking/bus.rs` |
| **Message** | `{eq_u_r: [F; D_EF], eq_u_r_omega: [F; D_EF], eq_u_r_prod: [F; D_EF]}` |

Carries the base-case equality polynomial evaluations from EqBaseAir. These three values represent:
- `eq_0(u, r)`: the base equality polynomial at `(u, r)`.
- `eq_0(u, r * omega)`: the same at a shifted point.
- `eq_0(u, 1) * eq_0(r, omega^{-1})`: a factored product form.

**Send set:** One message per proof.

**Producers:** EqBaseAir (send).
**Consumers:** SumcheckRoundsAir (receive).

**Invariants:**
- All three values are extension field elements.

---

#### 6.4.6 EqBitsInternalBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `stacking/bus.rs` |
| **Message** | `{b_value: F, num_bits: F, eval: [F; D_EF], child_lsb: F}` |

Internal recursive bus for the EqBits computation tree. Passes partial equality polynomial evaluations between tree levels, where each level processes one bit of the evaluation point.

**Send set:** One message per internal node in the binary evaluation tree.

**Producers:** EqBitsAir (send, from child to parent).
**Consumers:** EqBitsAir (receive, at parent node).

**Invariants:**
- `b_value` is the most significant `num_bits` bits of the row index (without LSB).
- `num_bits` = `n_stack - n_j - 1`.
- `eval` = `eq_{num_bits}(u_{> n_j}, b_j)`.
- `child_lsb` is the least significant bit of the receiving row's `b_value`.

---

#### 6.4.7 EqKernelLookupBus

| Property | Value |
|---|---|
| **Type** | Lookup, per-proof |
| **Source** | `stacking/bus.rs` |
| **Message** | `{n: F, eq_in: [F; D_EF], k_rot_in: [F; D_EF]}` |

Lookup table of equality kernel values indexed by dimension. Stores the `eq` and rotation kernel `k_rot` input values for each dimension `n`.

**Table:** For each dimension `n`, the `(eq_in, k_rot_in)` pair.

**Providers:** SumcheckRoundsAir, EqBaseAir (add_key_with_lookups).
**Consumers:** OpeningClaimsAir (lookup_key).

---

#### 6.4.8 EqBitsLookupBus

| Property | Value |
|---|---|
| **Type** | Lookup, per-proof |
| **Source** | `stacking/bus.rs` |
| **Message** | `{b_value: F, num_bits: F, eval: [F; D_EF]}` |

Lookup table of partial equality polynomial evaluations indexed by bit pattern. Allows stacking sumcheck AIRs to look up `eq_{num_bits}(u_{> n_j}, b_j)` for the appropriate bit decomposition.

**Table:** For each `(b_value, num_bits)` combination, the evaluated eq polynomial.

**Provider:** EqBitsAir (add_key_with_lookups).
**Consumers:** OpeningClaimsAir (lookup_key).

**Invariants:**
- `b_value` is the most significant `num_bits` bits of the row index.
- `num_bits` = `n_stack - n_j`.
- `eval` = `eq_{num_bits}(u_{> n_j}, b_j)`.

---

### 6.5 WHIR Module Internal Buses

#### 6.5.1 WhirSumcheckBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `whir/bus.rs` |
| **Message** | `{tidx: F, sumcheck_idx: F, pre_claim: [F; D_EF], post_claim: [F; D_EF]}` |

Carries sumcheck claims through WHIR rounds. Each WHIR round involves a sumcheck, and this bus connects the pre-claim (input) and post-claim (output) of each round's sumcheck.

**Send set:** One message per WHIR sumcheck round.

**Producers:** WhirRoundAir (send).
**Consumers:** SumcheckAir (WHIR) (receive).

**Invariants:**
- `sumcheck_idx` identifies which WHIR round this sumcheck belongs to.
- `pre_claim` is the claim entering the sumcheck; `post_claim` is the reduced claim after.

---

#### 6.5.2 WhirAlphaBus

| Property | Value |
|---|---|
| **Type** | Lookup, per-proof |
| **Source** | `whir/bus.rs` |
| **Message** | `{idx: F, challenge: [F; D_EF]}` |

Lookup table of WHIR alpha challenges. The alpha challenges are used for folding in the WHIR protocol.

**Table:** For each round index, the corresponding alpha challenge.

**Provider:** SumcheckAir (WHIR) (add_key_with_lookups).
**Consumers:** FinalPolyQueryEvalAir, WhirFoldingAir (lookup_key).

---

#### 6.5.3 WhirEqAlphaUBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `whir/bus.rs` |
| **Message** | `{value: [F; D_EF]}` |

Carries the product `eq(alpha, u)` from the WHIR round logic to the query verification. This is a single extension field value per proof.

**Send set:** One message per proof.

**Producers:** SumcheckAir (WHIR) (send).
**Consumers:** FinalPolyMleEvalAir (receive).

---

#### 6.5.4 VerifyQueriesBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `whir/bus.rs` |
| **Message** | `{tidx: F, whir_round: F, num_queries: F, omega: F, gamma: [F; D_EF], pre_claim: [F; D_EF], post_claim: [F; D_EF]}` |

Coordinates the query verification phase of each WHIR round. Carries the gamma challenge, the domain generator omega, and the pre/post claims around the query verification step.

**Send set:** One message per WHIR round that has queries.

**Producers:** WhirRoundAir (send).
**Consumers:** WhirQueryAir (receive).

**Invariants:**
- `num_queries` is the number of queries to verify in this round.
- `omega` is the domain generator for this WHIR round (squaring of the initial domain generator per round).
- `gamma` is the folding challenge for query verification.
- `pre_claim` and `post_claim` bracket the query verification.

---

#### 6.5.5 VerifyQueryBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `whir/bus.rs` |
| **Message** | `{whir_round: F, query_idx: F, merkle_idx_bit_src: F, zi_root: F, zi: F, yi: [F; D_EF]}` |

Carries individual query verification data. Each query involves opening a Merkle path and checking that the opened value is consistent with the WHIR folding.

**Send set:** One message per query per WHIR round.

**Producers:** WhirQueryAir (send).
**Consumers:** InitialOpenedValuesAir, NonInitialOpenedValuesAir (receive).

**Invariants:**
- `zi_root` and `zi` are evaluation domain points.
- `yi` is the claimed polynomial evaluation at `zi`.

---

#### 6.5.6 WhirQueryBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `whir/bus.rs` |
| **Message** | `{whir_round: F, query_idx: F, value: [F; D_EF]}` |

Carries the query evaluation results from the WHIR query verification AIR. Each message contains the accumulated value for a query in a specific round.

**Send set:** One message per query per WHIR round.

**Producers:** WhirQueryAir, WhirRoundAir (send).
**Consumers:** FinalPolyQueryEvalAir (receive).

---

#### 6.5.7 WhirGammaBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `whir/bus.rs` |
| **Message** | `{idx: F, challenge: [F; D_EF]}` |

Carries gamma challenges used in the WHIR query batching. Gamma is sampled once per WHIR round and used to batch multiple query results.

**Send set:** One message per WHIR round.

**Producers:** WhirRoundAir (send).
**Consumers:** FinalPolyQueryEvalAir (receive).

---

#### 6.5.8 WhirFoldingBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `whir/bus.rs` |
| **Message** | `{whir_round: F, query_idx: F, height: F, coset_shift: F, coset_idx: F, coset_size: F, twiddle: F, value: [F; D_EF], z_final: F, y_final: [F; D_EF]}` |

Carries intermediate folding state during the WHIR FRI-like folding process. Each message represents one step in the recursive folding of a coset, tracking the coset geometry and accumulated polynomial value.

**Send set:** One message per folding step per query per WHIR round.

**Producers:** InitialOpenedValuesAir, NonInitialOpenedValuesAir (send leaf values), WhirFoldingAir (send parent values in folding tree).
**Consumers:** WhirFoldingAir (receive child values at parent node).

**Invariants:**
- `coset_shift`, `coset_idx`, `coset_size` describe the coset geometry at this folding level.
- `twiddle` is the twiddle factor for this folding step.
- `value` is the accumulated polynomial evaluation.
- `z_final` and `y_final` carry the final evaluation point and value for verification.

---

#### 6.5.9 FinalPolyMleEvalBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `whir/bus.rs` |
| **Message** | `{tidx: F, num_whir_rounds: F, value: [F; D_EF]}` |

Carries the MLE evaluation of the final polynomial. After all WHIR rounds, the remaining polynomial is small enough to evaluate directly; this bus carries that evaluation.

**Send set:** One message per proof.

**Producers:** WhirRoundAir (send).
**Consumers:** FinalPolyMleEvalAir (receive).

**Invariants:**
- `num_whir_rounds` is the total number of WHIR rounds.
- `value` is the MLE evaluation of the final polynomial at the opening point.

---

#### 6.5.10 FinalPolyFoldingBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `whir/bus.rs` |
| **Message** | `{proof_idx: F, depth: F, node_idx: F, num_nodes_in_layer: F, value: [F; D_EF]}` |

Internal bus for the final polynomial's MLE evaluation tree folding. The MLE evaluation is computed as a binary tree of partial products, and this bus carries values between tree layers.

**Send set:** One message per node in the MLE evaluation tree.

**Producers:** FinalPolyMleEvalAir (send, from children to parent).
**Consumers:** FinalPolyMleEvalAir (receive, at parent).

**Invariants:**
- `depth` is the tree depth (0 = root).
- `node_idx` is the node's position within its layer.
- `num_nodes_in_layer` is the total nodes at this depth.
- Note: `proof_idx` appears as an explicit field in the message struct in addition to the per-proof prefix prepended by the bus macro.

---

#### 6.5.11 FinalPolyQueryEvalBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `whir/bus.rs` |
| **Message** | `{last_whir_round: F, value: [F; D_EF]}` |

Carries the final polynomial's evaluation at query points. After folding in the last WHIR round, the query must be checked against the final polynomial's direct evaluation.

**Send set:** One message per query in the final WHIR round.

**Producers:** WhirRoundAir (send).
**Consumers:** FinalPolyQueryEvalAir (receive).

**Invariants:**
- `last_whir_round` identifies the final WHIR round.
- `value` is the polynomial evaluation at the queried point.

---

#### 6.5.12 WhirFinalPolyBus

| Property | Value |
|---|---|
| **Type** | Lookup, per-proof |
| **Source** | `whir/bus.rs` |
| **Message** | `{idx: F, coeff: [F; D_EF]}` |

Lookup table of the final polynomial's coefficients. After all WHIR rounds of folding, the remaining polynomial has few enough coefficients to be sent directly; this bus makes those coefficients available.

**Table:** For each coefficient index, the extension field coefficient value.

**Provider:** FinalPolyMleEvalAir (add_key_with_lookups).
**Consumers:** FinalPolyQueryEvalAir (lookup_key).

---

## 7. Continuations Buses

These buses support the proof continuations feature, where verification state is carried across multiple aggregation steps.

### 7.1 CachedCommitBus

| Property | Value |
|---|---|
| **Type** | Permutation, per-proof |
| **Source** | `bus.rs` |
| **Message** | `{air_idx: F, cached_idx: F, cached_commit: [F; DIGEST_SIZE]}` |

Defined for the continuations protocol to carry cached trace commitments. When AIRs have cached (pre-committed) trace partitions, this bus is intended to verify that the cached commitment matches the expected value from the verifying key. Currently only partially wired: ProofShapeAir sends on this bus, but there is no corresponding receive in any AIR.

**Send set:** One message per cached trace partition per AIR.

**Producers:** ProofShapeAir (send).
**Consumers:** None currently wired (send-only from ProofShapeAir).

**Invariants:**
- `cached_commit` is a Poseidon2 digest of the cached trace.
- `cached_idx` identifies which cached partition within the AIR.

---

---

## Summary Table

| # | Bus Name | Type | Scope | Module |
|---|---|---|---|---|
| 1 | TranscriptBus | Permutation | Per-proof | Global |
| 2 | GkrModuleBus | Permutation | Per-proof | Global |
| 3 | BatchConstraintModuleBus | Permutation | Per-proof | Global |
| 4 | StackingModuleBus | Permutation | Per-proof | Global |
| 5 | WhirModuleBus | Permutation | Per-proof | Global |
| 6 | WhirMuBus | Permutation | Per-proof | Global |
| 7 | MerkleVerifyBus | Permutation | Per-proof | Global |
| 8 | Poseidon2PermuteBus | Lookup | Global | Crypto |
| 9 | Poseidon2CompressBus | Lookup | Global | Crypto |
| 10 | AirShapeBus | Lookup | Per-proof | Data/Shape |
| 11 | HyperdimBus | Lookup | Per-proof | Data/Shape |
| 12 | LiftedHeightsBus | Lookup | Per-proof | Data/Shape |
| 13 | CommitmentsBus | Lookup | Per-proof | Data/Shape |
| 14 | PublicValuesBus | Permutation | Per-proof | Data/Shape |
| 15 | StackingIndicesBus | Lookup | Per-proof | Data/Shape |
| 16 | ColumnClaimsBus | Permutation | Per-proof | Data/Shape |
| 17 | XiRandomnessBus | Permutation | Per-proof | Challenge |
| 18 | ConstraintSumcheckRandomnessBus | Permutation | Per-proof | Challenge |
| 19 | WhirOpeningPointBus | Permutation | Per-proof | Challenge |
| 20 | WhirOpeningPointLookupBus | Lookup | Per-proof | Challenge |
| 21 | EqNegBaseRandBus | Permutation | Per-proof | Challenge |
| 22 | RangeCheckerBus | Lookup | Global | Primitives |
| 23 | PowerCheckerBus | Lookup | Global | Primitives |
| 24 | ExpBitsLenBus | Lookup | Global | Primitives |
| 25 | SelUniBus | Lookup | Per-proof | Computation |
| 26 | SelHypercubeBus | Lookup | Per-proof | Computation |
| 27 | EqNegResultBus | Permutation | Per-proof | Computation |
| 28 | ExpressionClaimNMaxBus | Permutation | Per-proof | Computation |
| 29 | FractionFolderInputBus | Permutation | Per-proof | Computation |
| 30 | NLiftBus | Permutation | Per-proof | Computation |
| 31 | ProofShapePermutationBus | Permutation | Per-proof | ProofShape Internal |
| 32 | StartingTidxBus | Permutation | Per-proof | ProofShape Internal |
| 33 | NumPublicValuesBus | Permutation | Per-proof | ProofShape Internal |
| 34 | GkrXiSamplerBus | Permutation | Per-proof | GKR Internal |
| 35 | GkrLayerInputBus | Permutation | Per-proof | GKR Internal |
| 36 | GkrLayerOutputBus | Permutation | Per-proof | GKR Internal |
| 37 | GkrSumcheckInputBus | Permutation | Per-proof | GKR Internal |
| 38 | GkrSumcheckOutputBus | Permutation | Per-proof | GKR Internal |
| 39 | GkrSumcheckChallengeBus | Permutation | Per-proof | GKR Internal |
| 40 | BatchConstraintConductorBus | Lookup | Per-proof | BatchConstraint Internal |
| 41 | SumcheckClaimBus | Permutation | Per-proof | BatchConstraint Internal |
| 42 | SymbolicExpressionBus | Lookup | Per-proof | BatchConstraint Internal |
| 43 | ExpressionClaimBus | Permutation | Per-proof | BatchConstraint Internal |
| 44 | InteractionsFoldingBus | Permutation | Per-proof | BatchConstraint Internal |
| 45 | ConstraintsFoldingBus | Permutation | Per-proof | BatchConstraint Internal |
| 46 | Eq3bBus | Permutation | Per-proof | BatchConstraint Internal |
| 47 | EqSharpUniBus | Permutation | Per-proof | BatchConstraint Internal |
| 48 | EqZeroNBus | Permutation | Per-proof | BatchConstraint Internal |
| 49 | EqNOuterBus | Lookup | Per-proof | BatchConstraint Internal |
| 50 | EqNegInternalBus | Permutation | Per-proof | BatchConstraint Internal |
| 51 | UnivariateSumcheckInputBus | Permutation | Per-proof | BatchConstraint Internal |
| 52 | StackingModuleTidxBus | Permutation | Per-proof | Stacking Internal |
| 53 | ClaimCoefficientsBus | Permutation | Per-proof | Stacking Internal |
| 54 | SumcheckClaimsBus | Permutation | Per-proof | Stacking Internal |
| 55 | EqRandValuesLookupBus | Lookup | Per-proof | Stacking Internal |
| 56 | EqBaseBus | Permutation | Per-proof | Stacking Internal |
| 57 | EqBitsInternalBus | Permutation | Per-proof | Stacking Internal |
| 58 | EqKernelLookupBus | Lookup | Per-proof | Stacking Internal |
| 59 | EqBitsLookupBus | Lookup | Per-proof | Stacking Internal |
| 60 | WhirSumcheckBus | Permutation | Per-proof | WHIR Internal |
| 61 | WhirAlphaBus | Lookup | Per-proof | WHIR Internal |
| 62 | WhirEqAlphaUBus | Permutation | Per-proof | WHIR Internal |
| 63 | VerifyQueriesBus | Permutation | Per-proof | WHIR Internal |
| 64 | VerifyQueryBus | Permutation | Per-proof | WHIR Internal |
| 65 | WhirQueryBus | Permutation | Per-proof | WHIR Internal |
| 66 | WhirGammaBus | Permutation | Per-proof | WHIR Internal |
| 67 | WhirFoldingBus | Permutation | Per-proof | WHIR Internal |
| 68 | FinalPolyMleEvalBus | Permutation | Per-proof | WHIR Internal |
| 69 | FinalPolyFoldingBus | Permutation | Per-proof | WHIR Internal |
| 70 | FinalPolyQueryEvalBus | Permutation | Per-proof | WHIR Internal |
| 71 | WhirFinalPolyBus | Lookup | Per-proof | WHIR Internal |
| 72 | CachedCommitBus | Permutation | Per-proof | Continuations |
