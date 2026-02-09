# Variable Range Checker

This chip is similar in functionality to the [Range Checker](../range/README.md) but is more general. It is initialized with a `range_max_bits` value and provides a lookup table for range checking a variable `x` has `b` bits where `b` can be any integer in `[0, range_max_bits]`. In other words, this chip can be used to range check for different bit sizes. We define `0` to have `0` bits.

Conceptually, this works like `range_max_bits` different lookup tables stacked together:
- One table for 1-bit values
- One table for 2-bit values
- And so on up to `range_max_bits`-bit values

With the `max_bits` column indicating which bit-size to check against.

For example, with `range_max_bits = 3`, the lookup table contains:
- All 1-bit values: 0, 1
- All 2-bit values: 0, 1, 2, 3
- All 3-bit values: 0, 1, 2, 3, 4, 5, 6, 7

The chip uses gate-based constraints to generate the trace columns instead of a preprocessed trace. The trace enumerates all valid `(value, max_bits)` pairs in a specific order: for each bit size `b` from 0 to `range_max_bits`, it enumerates all values from 0 to $2^b - 1$. The order is: `[0,0]`, `[0,1]`, `[1,1]`, `[0,2]`, `[1,2]`, `[2,2]`, `[3,2]`, `[0,3]`, ...

**Columns:**
- `value`: The value being range checked
- `max_bits`: The maximum number of bits for this value
- `two_to_max_bits`: Helper column storing $2^{\mathtt{max\_bits}}$
- `mult`: Multiplicity column tracking how many range checks are requested for each `(value, max_bits)` pair

The constraints enforce the enumeration pattern by observing that `value + two_to_max_bits` equals `row_index + 1` (a strictly increasing sequence). By constraining this sum to increase by exactly 1 each row, combined with constraints that `value` can only be 0 or increment and `max_bits` can only stay or increment, the trace is forced into the correct enumeration:
1. First row: start at `[value=0, max_bits=0, two_to_max_bits=1]`
2. Transitions: `max_bits` can only stay the same or increment by 1; `value` can only be 0 or increment by 1; `two_to_max_bits` doubles when `max_bits` increments; and `value + two_to_max_bits` increases by exactly 1 each row
3. Last row: end at `[value=0, max_bits=range_max_bits+1, mult=0]` (dummy row to make trace height a power of 2)

The last-row constraint acts as a checksum—if the trace ever "cheats" (e.g., `value` continues past $2^{\mathtt{max\_bits}} - 1$ instead of wrapping to 0), it cannot reach the required final state.

The functionality and usage of the chip are very similar to those of the [Range Checker](../range/README.md) chip.
