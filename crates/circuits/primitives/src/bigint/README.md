# BigInt modular arithmetic by checking carry to zero

## Context

In general we need to work with BigUints, which we say consist of representations via `limbs: [F; num_limbs]` where the unsigned integer equals `\sum_i limbs[i] * 2^{limb_bits * i}`. In the canonical form we assume `limbs[i]` in `[0, 2^limb_bits)` (unsigned `limb_bits` bits). 

The key optimization is to allow representations of BigInts as `OverflowInts` with `limbs[F; num_limbs]`
where `limbs[i]` in `[-2^overflow_bits, 2^overflow_bits)`.
The integer equals `sum_i limbs[i] * 2^{limb_bits * i}`. We must have `overflow_bits < 30`. 

The core functionality that is needed is `check_carry_to_zero`: which is the constraint that given OverflowInt limbs, you can constrain the corresponding integer equals 0. This is done by a sequence of carries to get the overflow integer into canonical form (and check it’s zero). The carries must be range checked to be limb_bits. 

The use case is commonly if we have multiple OverflowInts `a,b,p,q,r` we may want to check 
`check_carry_to_zero(a * b + p * q - r)` where `a * b` is expanded bigint multiplication, but without carries. Observe this proves `a * b = r (mod p)`. 

We decided that `check_carry_to_zero` should not be a separate AIR that receives interactions:
the reason is that you would need to receive all the limbs on this AIR which duplicates the total number of cells by quite a lot. 
We also discussed that `limb_bits` should be ~10 to handle 256-bit integers, and it is not worth using a large `limb_bits` and then decomposing to smaller chunks for the carries, because then you need to do further range checks on the decomposition.

## Implementation details

### Assumptions
* Inputs `x, y` are of the same length (when represented as limbs) as `p`. They don't necessarily be within `[0, p)`.

### `OverflowInt<T>`

It's big integer represented as limbs, and also tracks the value limit of each limb.
- it supports arithmetic like +-*, and updates the value limit accordingly.
- the generic type can be real values like `isize` or expression like `AB::Expr`.
- can generate the carries array for the limbs automatically.

Use case:

1. Trace generation: The parent AIR (e.g. multiplication AIR) can compose the expression (e.g. AB - PQ - R) as `OverflowInt<isize>` and generates the carries array (see the tests for an example).
1. AIR eval: The parent AIR `eval` function should compose the expression as `OverflowInt<AB::Expr>` and pass it to the check carry subair `constrain_carry_to_zero` along with the carries column.

### Range checker bit v.s. OverflowInt bit limit

There is a subtle difference between the `max_overflow_bits` tracked within `OverflowInt`, and the bit we range check.

`OverflowInt` starts from something nonnegative like a `BigUint` or `AB::Var`,
and we keep track of the maximum absolute value of each element (and the `max_overflow_bits = ceil_log2(max_abs)` ).
Each supported operation (namely +-*) updates the max abs value.

Given an OverflowInt with `max_overflow_bits = m`, and assume the non-overflow limb bits is `k`.
The positive carry is at most `m - k` bits. And since the negative carry is rounding down (see below section) it’s at most (in the abs sense) `-2^(m-k)`.
So the range of carry should be `[-2^(m-k), 2^(m-k))`. The way we range-check this is by adding `2^(m-k)` to it so it’s within `[0, 2^(m-k+1))` and this within `m-k+1` bits.

### Negative carries

Assume the limb bit (not overflowed) is `k`, and one of the expr of `OverflowInt` is evaluated to `X = -( 2^k * a + b)`.
If `b = 0` (which is usually the case as this subair is used to constrain something evaluated to 0), then carry is exactly `-a` without worrying about taking floor or ceil.
However, to make it complete let’s assume b is not 0. Since the canonical limb is nonnegative within `[0, 2^k - 1)`,
the carry should be `-(a+1)` with canonical limb `2^k - b`. 
Therefore the correct way to calculate carry is `X >> k` instead of `X / (1 << K)`. 

```Rust
fn main() {
    let val: isize = 63;
    assert_eq!(val >> 2, 15);
    assert_eq!(val / 4, 15);
    let val2: isize = -63;
    assert_eq!(val2 >> 2, -16);  // >> round toward -inf
    assert_eq!(val2 / 4, -15);   // / round toward 0
}
```

### Different AIRs for each operation v.s. single AIR for all operations

Assuming the modulus prime is of N limbs.
For multiplication, the columns are: `x, y, q, r` of size N, and `carries` of size 2N-1 (as `x*y` can be of length 2N-1). 
Division is just a different equation: `yr - x - pq = 0`, so the same columns work.
For addition and subtractions: `x [+/-] y - r - pq = 0` , the main difference here is that we know that `q` will just be one limb,
and thus the equation should have limb size N, and `carries` just be of size N.
If we combine the operations into one chip, we will also need extra columns: `opcode_add_flag, opcode_sub_flag. opcode_mul_flag , opcode_div_flag`.
Therefore we will just have separate chips for different operations.

We are unsure if division is actually needed, so commented out for now.

## Note on `is_valid` boolean check

Both the `CheckCarryModToZeroSubAir` and `CheckCarryToZeroSubAir` subair's do not assert that `is_valid` is boolean.
They assume the parent air already does this.

This is to avoid duplicating the `is_valid` boolean check every time we use these subair's, since we may call `CheckCarryModToZeroSubAir::eval` multiple times in the parent air's `eval` method.