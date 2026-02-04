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

The chip uses gate-based constraints to generate the trace columns instead of a preprocessed trace. The trace enumerates all valid `(value, max_bits)` pairs in a specific order: for each bit size `b` from 0 to `range_max_bits`, it enumerates all values from 0 to $2^b - 1$. The order is: `[0,0]`, `[0,1]`, `[1,1]`, `[0,2]`, `[1,2]`, `[2,2]`, `[3,2]`, `[0,3]`, ...

**Columns:**
- `value`: The value being range checked
- `max_bits`: The maximum number of bits for this value
- `two_to_max_bits`: Helper column storing $2^{\mathtt{max\_bits}}$, used to detect wrap transitions
- `selector_inverse`: The inverse of the selector `(value + 1 - two_to_max_bits)`, used to create a boolean selector for wrap detection
- `is_not_wrap`: Boolean selector (1 if NOT wrapping, 0 if wrapping), used to reduce degree of transition constraints
- `mult`: Multiplicity column tracking how many range checks are requested for each `(value, max_bits)` pair

The constraints enforce the enumeration pattern by:
1. Starting at `[0, 0]` (first-row constraints)
2. Detecting wrap transitions when `value` reaches $2^{\mathtt{max\_bits}} - 1$ using the selector-based detection
3. Enforcing correct progression: increment `value`, or wrap to `[0, max_bits+1]`
4. Ending with a dummy row `[0, range_max_bits+1]` to make trace height a power of 2, constraining this row to have mult=0

The functionality and usage of the chip are very similar to those of the [Range Checker](../range/README.md) chip.
