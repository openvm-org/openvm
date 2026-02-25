# Recursion Range Checker AIR 

## 1. Executive summary

`RangeCheckerAir<NUM_BITS>` is a lookup-table AIR that gets requests to verify that a number is in the range `[0, 2^NUM_BITS)`, by materializing values in `\[0, 2^NUM_BITS)` and sends them to the `RangeCheckerBus`. This guarantees that as long as this bus is balanced, the requested values are within `NUM_BITS` of bits.

## 2. Public values

This AIR has no public values in its trace interface.

## 3. Interface, assumptions, and guarantees

### 3.1 Bus interface

- Receives: none.
- Sends: on every row, one `add_key_with_lookups` on `RangeCheckerBus` with key
  `RangeCheckerBusMessage { value: local.value, max_bits: NUM_BITS }`
  and multiplicity `local.mult`.

### 3.2 Assumptions

- Total number of multiplicity should not overflow the field size.

### 3.3 Guarantees to the rest of the system

The `value` are all within `NUM_BITS` of bits.

## 4. Proof

The statement is: the `value` column is exactly `0,1,2,...,2^NUM_BITS-1`.

This is guaranteed by the constraints that:
1. The first row has `value = 0`
2. For all the transitions, `next.value = local.value + 1`
3. The last row has `value = 2^NUM_BITS -1`

## 5. Example

For `max_numbits = 3`, the trace has `2^3 = 8` rows, and `value` must be:

| value | mult (example) |
| --- | --- |
| 0 | 2 |
| 1 | 0 |
| 2 | 1 |
| 3 | 3 |
| 4 | 0 |
| 5 | 1 |
| 6 | 0 |
| 7 | 4 |

Only `value` is constrained by transition rules; `mult` is the lookup multiplicity for that key.
