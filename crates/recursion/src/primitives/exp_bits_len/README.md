# Recursion ExpBitsLen AIR

## Summary

`ExpBitsLenAir` serves lookups of the form

`(base, bit_src, num_bits, result)`

where

`result = base^(bit_src mod 2^num_bits)`

using the canonical BabyBear representative of `bit_src`.

This primitive is shared by the recursion PoW and query AIRs so they can reuse the same low-bit exponent logic.

This AIR is BabyBear-specific. Its canonicality check is hard-coded to the BabyBear modulus
`p = 15 * 2^27 + 1`, and the implementation will panic if instantiated over a different
`PrimeField32`.

## Trace shape

Each request always occupies exactly `32` rows:

- rows `0..30` are the 31 decomposition steps
- row `31` is the terminal state after the final shift

The request row is row `0`. Only that row publishes the external lookup key on `ExpBitsLenBus`.

Across a block:

- `base` squares each step
- `bit_src` shifts right by one bit each step
- `num_bits` decrements until it reaches `0`, then stays `0`
- `result` updates only while `num_bits > 0`
- decomposition continues all the way to the terminal row even after `num_bits == 0`

So the external meaning stays the same, but the underlying bit decomposition is fully constrained.

The AIR stores `is_first`, but does not store `is_last` as a column. Instead, on a local/next row
pair it derives:

- `is_transition = next.is_valid - next.is_first`
- `local_is_last = local.is_valid - next.is_valid + next.is_first`

This gives a fixed 32-row block structure while still handling exact-height traces and a padding
tail with the same formula.

## Canonicality check

Decomposing all 31 bits is not enough by itself in BabyBear, because

`BabyBear::ORDER_U32 = 0x78000001 < 2^31`.

That means some field elements also admit a second 31-bit representative as `x + p`.

For BabyBear we can rule that out with a low-degree check because

`p = 15 * 2^27 + 1`.

A 31-bit integer is `>= p` iff:

- its top four bits `b27..b30` are all `1`
- and at least one of the low 27 bits `b0..b26` is `1`

The AIR therefore carries two running booleans:

- `low_bits_are_zero`: all bits `b0..b26` seen so far are zero
- `high_bits_all_one`: all high bits `b27..b30` seen so far are one

On the terminal row it enforces:

`high_bits_all_one * (1 - low_bits_are_zero) = 0`

which excludes the non-canonical `x + p` decomposition while still allowing `p - 1`.

## Main constraints

The AIR enforces:

1. `bit_src_mod_2` is boolean and adjacent rows satisfy `bit_src = 2 * next.bit_src + bit_src_mod_2`
2. `num_bits` and `low_bits_left` imply their boolean “nonzero” flags via `when(counter).assert_one(selector)`, and the fixed decrement-to-zero transitions force those selectors back to `0` once the counters reach `0`
3. `result_multiplier` is either `1` or `base`, depending on whether the current bit is active
4. `is_first => is_valid`, and adjacent rows enforce the fixed 32-row block transition through the derived `is_transition` / `local_is_last` selectors above
5. the terminal row enforces `bit_src = 0`, `num_bits = 0`, and `result = 1`
6. the terminal row also enforces the canonical `< p` condition above

These constraints ensure that:

- the bit decomposition is fully constrained through the terminal row
- `result` is computed from the requested low `num_bits` bits
- `bit_src` is interpreted using the canonical BabyBear representative

Padding rows do not publish interactions. The AIR does not require a unique padding-row encoding;
it only constrains the selector structure needed to separate valid request blocks from padding.
