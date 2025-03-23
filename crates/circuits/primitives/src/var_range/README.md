# Variable Range Checker

This chip is similar in functionality to the [Range Checker](../range/README.md) but is more general. It is initialized with a `range_max_bits` value and provides a lookup table for range checking a variable `x` has `b` bits where `b` can be any integer in `[0, range_max_bits]`. In other words, this chip can be used to range check for different bit sizes. We define `0` to have `0` bits.

Conceptually, this works like `range_max_bits` different lookup tables stacked together:
- One table for 1-bit values
- One table for 2-bit values
- And so on up to `range_max_bits`-bit values

With a selector column indicating which bit-size to check against.

For example, with `range_max_bits = 3`, the lookup table contains:
- All 1-bit values: 0, 1
- All 2-bit values: 0, 1, 2, 3
- All 3-bit values: 0, 1, 2, 3, 4, 5, 6, 7

The chip has three columns `value, max_bits, mult`. The `value, max_bits` columns are preprocessed and the `mult` column is left unconstrained. The `(value, max_bits)` preprocessed trace is populated with `(x, b)` for all $x \in [0, 2^b)$ and $b \in [0, \mathtt{range\\_max\\_bits}]$.

**Preprocessed Columns:**
- `value`: The value being range checked
- `max_bits`: The maximum number of bits for this value

**IO Columns:**
- `mult`: Multiplicity column tracking how many range checks are requested for each (value, max_bits) pair

The functionality and usage of the chip are very similar to those of the [Range Checker](../range/README.md) chip.
