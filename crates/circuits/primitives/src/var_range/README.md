# Variable Range Checker

This chip is similar in functionality to the [Range Checker](../range/README.md) but is more general. It is initialized with a `range_max_bits` value and provides a lookup table for range checking a variable `x` has `b` bits where `b` can be any integer in `[0, range_max_bits]`. In other words, this chip can be used to range check for different bit sizes. We define `0` to have `0` bits.

The chip has three columns `value, max_bits, mult`. The `value, max_bits` columns are preprocessed and the `mult` column is left unconstrained. The `(value, max_bits)` preprocessed trace is populated with `(x, b)` for all $x \in [0, 2^b)$ and $b \in [0, \mathtt{range\\_max\\_bits}]$.

The functionality and usage of the chip are very similar to those of the [Range Checker](../range/README.md) chip.
