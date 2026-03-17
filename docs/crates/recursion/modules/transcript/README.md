# Transcript Module

**AIRs (3):** TranscriptAir, Poseidon2Air, MerkleVerifyAir.
**Per-AIR details:** [Transcript AIRs](./airs.md) (trace columns, walkthroughs, bus summaries).

## Interface

The Transcript module provides three services to the rest of the circuit:

1. **Fiat-Shamir transcript** (`TranscriptBus`, permutation). TranscriptAir **sends**
   `(tidx, value, is_sample)` messages; all other AIRs **receive** them. This is the
   primary interface — every observe/sample in the verification protocol flows through it.

2. **Poseidon2 hash evaluation** (`Poseidon2PermuteBus`, `Poseidon2CompressBus`, both
   lookup). Poseidon2Air provides tables of Poseidon2 permutation and compression
   evaluations. TranscriptAir uses permute lookups for sponge transitions; MerkleVerifyAir
   uses compress lookups for Merkle path hashing; InitialOpenedValuesAir uses permute
   lookups for row hashing; NonInitialOpenedValuesAir uses compress lookups for value
   hashing.

3. **Merkle path verification** (`MerkleVerifyBus`, permutation). MerkleVerifyAir verifies
   Merkle authentication paths. WHIR's InitialOpenedValuesAir and NonInitialOpenedValuesAir
   send leaf hashes on this bus; MerkleVerifyAir walks up the authentication path using
   `Poseidon2CompressBus` at each level, then verifies the root against `CommitmentsBus`
   (provided by ProofShapeAir for stacking commitments and WhirRoundAir for round
   commitments). MerkleVerifyAir uses `RightShiftBus` (from ExpBitsLenAir) to derive the
   authentication-path index from the raw Merkle index.

## Contract

For each proof, TranscriptAir sends exactly one `TranscriptBus` message per
`tidx` value (0, 1, 2, ...), each with a definite `(value, is_sample)`. The values are
determined by a Poseidon2 sponge starting from the zero state: if `is_sample_i = 0` then
`value_i` is absorbed into the rate, and if `is_sample_i = 1` then `value_i` equals the
sponge output at that position.

TranscriptAir does **not** constrain which positions are samples vs. observes — the other
AIRs fix that via their hardcoded `is_sample` in the bus receive.

**Consequence:** every sample value is uniquely determined by the sequence of all preceding
`(value_i, is_sample_i)` tuples and the Poseidon2 sponge.

### Sponge correctness

TranscriptAir constrains that the Fiat-Shamir sponge is executed correctly:

1. **Zero-state initialization.** Each proof's sponge starts with rate and capacity all
   zeros.
2. **Absorb/squeeze state machine.** The sponge processes values in chunks of CHUNK (= 8,
   the Poseidon2 rate). Same-mode continuations (absorb→absorb or squeeze→squeeze) require
   exactly CHUNK operations per row, ensuring a full chunk is consumed before permuting and
   continuing. Mode switches (absorb→squeeze, squeeze→absorb) may happen with a partial
   chunk.
3. **Permutation lookups.** Each `prev_state → post_state` transition is verified via
   `Poseidon2PermuteBus`. Permutations fire on absorb→squeeze transitions and when a full
   chunk completes; when transitioning from squeeze back to absorb the state is carried
   forward without permutation.
4. **State continuity.** The capacity portion of the sponge state is preserved across rows.
   Inactive rate lanes and squeeze-phase rows carry state forward unchanged.
5. **Squeeze output ordering.** During squeeze, output values are read from the rate in
   reverse lane order (`prev_state[CHUNK-1-i]`).
6. **Final state export.** At the end of each proof's transcript, the final sponge state
   is sent to `FinalTranscriptStateBus` for use by downstream verification stages.

### Merkle verification correctness

MerkleVerifyAir establishes commitment-opening correctness end-to-end:

1. **Leaf binding.** InitialOpenedValuesAir and NonInitialOpenedValuesAir send leaf hashes on
   `MerkleVerifyBus`, so the Merkle verifier is pinned to the opened rows / opened values
   chosen by the WHIR AIRs.
2. **Hash correctness.** Every parent digest used in the authentication path is checked by a
   `Poseidon2CompressBus` lookup, so each compression step is a valid Poseidon2 hash of its
   two children.
3. **Path continuity.** MerkleVerifyAir self-sends intermediate parents on
   `MerkleVerifyBus`, threading the same Merkle proof from the leaf-combining phase up to
   the root row.
4. **Root authentication.** On the root row, MerkleVerifyAir looks up
   `(commit_major, commit_minor, root_digest)` on `CommitmentsBus`. Those commitment entries
   are provided by ProofShapeAir for stacking commitments and by WhirRoundAir for WHIR round
   commitments.

Therefore an accepted Merkle opening is simultaneously tied to the prover-supplied leaves,
to correct Poseidon2 compression steps, and to a commitment that is already fixed elsewhere
in the proof.

**Bus dependencies:** `Poseidon2PermuteBus`, `Poseidon2CompressBus` (lookup tables from
Poseidon2Air); `CommitmentsBus` (root authentication lookup); `MerkleVerifyBus`
(permutation bus for leaf and intermediate hash flow); `RightShiftBus` (lookup for index
derivation).

**See also:** [Transcript AIRs](./airs.md) for per-AIR trace columns, walkthroughs, and bus summaries.
