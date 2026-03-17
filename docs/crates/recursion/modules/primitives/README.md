# Primitive Lookup Tables

**Per-AIR details:** [Primitive AIRs](./airs.md) (trace columns, walkthroughs).

These AIRs provide lookup tables used by other modules. They do not participate in the
module handoff bus chain. Their correctness is independent of other modules.

A lookup forces queried keys to exist in the table, but the table entries themselves are
prover-controlled (see [README.md](../../README.md#bus-semantics)). Each table AIR's contract
states that its constraints force all entries to be valid.

## Contracts

**Cryptographic providers:**
- **Poseidon2Air** (`Poseidon2PermuteBus`, `Poseidon2CompressBus`): every table entry is a
  valid Poseidon2 permutation or compression evaluation. The AIR delegates to
  `Poseidon2SubAir`, which constrains each round transition (S-box, linear layer, round
  constant addition) so that `output = Poseidon2(input)` for every row.

**Arithmetic providers:**
- **PowerCheckerAir** (`PowerCheckerBus`): every table entry satisfies `exp = 2^log`. The
  AIR enumerates values sequentially: `log` starts at 0 and increments by 1 each row, while
  `exp` starts at 1 and doubles (`exp' = exp * 2`), so only the geometric progression
  `{(0,1), (1,2), (2,4), ...}` can appear.
- **ExpBitsLenAir** (`ExpBitsLenBus`, `RightShiftBus`): every `ExpBitsLenBus` table entry
  satisfies `result = base^(bit_src & ((1 << num_bits) - 1))` — the base raised to the low
  `num_bits` bits of `bit_src`. The AIR uses recursive binary exponentiation: each row
  squares the base, extracts the current bit of `bit_src`, and conditionally multiplies the
  running result, so only correct partial exponentiations can be published. As a secondary
  table, `RightShiftBus` provides `result = input >> shift_bits`, derived from the same
  recursive structure. Used by MerkleVerifyAir for Merkle index derivation.

**Range/bit providers:**
- **RangeCheckerAir** (`RangeCheckerBus`): every table entry is in `[0, 2^NUM_BITS)`.
  `NUM_BITS` is a const generic parameter (instantiated as 8 in the verifier subcircuit).
  The AIR exhaustively enumerates the range: `value` starts at 0, increments by 1 each row,
  and ends at `2^NUM_BITS - 1`, so any looked-up value must match a row in this complete
  list.

These claims compose with the module extraction arguments: when a pipeline module looks up a
value, the lookup guarantees the value is in the table, and the table contract guarantees
the entries are correct.
