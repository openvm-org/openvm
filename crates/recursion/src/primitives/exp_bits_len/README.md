# Recursion ExpBitsLen AIR 

## 1. Executive summary

The job of `ExpBitsLenAir` is to provide a lookup service for messages
`(base, bit_src, num_bits, result)` where:

`result` equals `base` raised to the value of the lowest `num_bits` bits of `bit_src`.

In formula form:

`result = base^(bit_src mod 2^num_bits)`.

Why we need it: many AIRs (PoW checks and query exponent checks) can reuse this one bus lookup instead of implementing exponent logic repeatedly.

## 2. Public values

This AIR has no public values in its trace interface.

## 3. Example

### 3.1 Row count per request

Other AIRs can make a request: `add_request(&self, base: F, bit_src: F, num_bits: usize)`
to lookup the exp result.

For one request with `num_bits = n`, tracegen creates exactly `n + 1` valid rows.

- row `0` has `num_bits = 0` (base case, `result = 1`)
- row `n` has `num_bits = n` (the requested top key)

### 3.2 Relation between adjacent rows

- the next row tracks one more bit than the previous row (so `num_bits` increases by one)
- the next row uses the previous row's base before squaring (equivalently, the next base is one square-root step of the previous base)
- the next row's `sub_result` is exactly the previous row's `result`
- the next row computes `result` from `sub_result` and the current least-significant bit:
  - if the current low bit is `1`, multiply by `base`
  - if the current low bit is `0`, keep `sub_result` unchanged

This is exactly the recursive lookup rule encoded in AIR.

### 3.3 Concrete example

Example request: `base = g`, `bit_src = 45`, `num_bits = 4`.

`101101b` means binary (`b` = binary), so `101101b = 45` in decimal.

Because `num_bits = 4`, we use only the lowest 4 bits of `bit_src`:  
`45 = 101101b`, lowest 4 bits are `1101b = 13`, so exponent is `13`.


| base | bit_src | num_bits | bit_src_mod_2 | sub_result | result |
| --- | --- | --- | --- | --- | --- |
| `g^16` | 2 | 0 | 0 | 1 | 1 |
| `g^8` | 5 | 1 | 1 | 1 | `g^8` |
| `g^4` | 11 | 2 | 1 | `g^8` | `g^12` |
| `g^2` | 22 | 3 | 0 | `g^12` | `g^12` |
| `g` | 45 | 4 | 1 | `g^12` | `g^13` |

Top requested key is `(g, 45, 4, g^13)`, matching `g^(45 mod 16) = g^13`.

Note: there are additional auxiliary columns not shown in the table.

## 4. Proof

The statement is: for each valid row key `(base, bit_src, num_bits, result)`, `result` is the exponent over the low `num_bits` bits of `bit_src`.

This is guaranteed by constraints:

1. `bit_src_mod_2` is boolean, and `bit_src = 2 * bit_src_div_2 + bit_src_mod_2`
2. `num_bits = num_bits^2 * num_bits_inv`  
   so `num_bits * num_bits_inv` is `0` when `num_bits=0`, else `1`
3. For nonzero `num_bits`, lookup the subproblem at
   `(base^2, bit_src_div_2, num_bits-1, sub_result)`
4. For nonzero `num_bits`, enforce
   `result = sub_result * (bit_src_mod_2 * base + 1 - bit_src_mod_2)`. That is, if `bit_src_mod_2` is 1 we multiply sub_result by base. Otherwise do nothing. 
5. For zero `num_bits`, enforce `result = 1`

The constraints are all "local", and the relationship between adjacent rows is guaranteed by the self lookup: Every row `add_key_with_lookups`, and every row except first row `lookup_key` for the "previous row".

