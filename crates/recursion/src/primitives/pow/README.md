# Recursion Power Checker AIR 

## 1. Executive summary

The job of `PowerCheckerAir<BASE, N>` is to provide a lookup table for two common checks used by other AIRs:
1. power check: verify `exp = BASE^log`
2. range check: verify `log` is in `[0, N)`

It materializes `(log, pow)` where `pow = BASE^log` for `log in [0, N)`, sends those keys to `PowerCheckerBus`, and also sends `(value = log, max_bits = log2(N))` to `RangeCheckerBus`.

Note: these are two independent services. In trace generation, `add_pow(log)` (power check) and `add_range(value)` (range check) are independent APIs, so `mult_pow` and `mult_range` are tracked separately and are not required to be equal. 

## 2. Public values

This AIR has no public values in its trace interface.

## 3. Interface, assumptions, and guarantees

### 3.1 Bus interface

- Receives: none.
- Sends (per row):
  - `PowerCheckerBusMessage { log, exp: pow }` with multiplicity `mult_pow`
  - `RangeCheckerBusMessage { value: log, max_bits: log2(N) }` with multiplicity `mult_range`

### 3.2 Assumptions

- `N` is a power of two.
- Total multiplicities should not overflow field capacity.

### 3.3 Guarantees to the rest of the system

- `log` is exactly all values in `[0, N)`.
- `pow` is exactly `BASE^log` for each row.
- So the AIR supplies exactly:
  - `(log, BASE^log)` on `PowerCheckerBus`
  - `(log, log2(N))` on `RangeCheckerBus`

## 4. Proof

The statement is:

- `log` column is exactly `0,1,2,...,N-1`
- `pow` column is exactly `1,BASE,BASE^2,...,BASE^(N-1)`, so equal to `BASE^(log)`.

This is guaranteed by constraints:

1. First row: `log = 0`, `pow = 1`
2. Transition: `next.log = local.log + 1`
3. Transition: `next.pow = local.pow * BASE`
4. Last row: `log = N - 1`

So the only valid trace is the full sequence from `0` to `N-1` with matching powers.  
Then each row sends the two bus keys from Section 3.1.

## 5. Example

For `BASE = 2`, `N = 8`:

| log | pow | mult_pow (example) | mult_range (example) |
| --- | --- | --- | --- |
| 0 | 1 | 1 | 2 |
| 1 | 2 | 0 | 1 |
| 2 | 4 | 3 | 0 |
| 3 | 8 | 1 | 1 |
| 4 | 16 | 0 | 0 |
| 5 | 32 | 2 | 2 |
| 6 | 64 | 0 | 1 |
| 7 | 128 | 1 | 0 |

Only `log` and `pow` are constrained by transition rules; `mult_pow`/`mult_range` are lookup multiplicities.
