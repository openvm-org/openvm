# Short Weierstrass (SW) Curve Operations

The `ec_add_ne`, `ec_double`, and `ec_mul` instructions are implemented in the `weierstrass_chip` module.

### 1. `ec_add_ne`

**Assumptions:**

- Both points `(x1, y1)` and `(x2, y2)` lie on the curve and are not the identity point.
- `x1` and `x2` are distinct in the coordinate field.

**Circuit statements:**

- The chip takes two inputs: `(x1, y1)` and `(x2, y2)`, and returns `(x3, y3)` where:
  - `lambda = (y2 - y1) / (x2 - x1)`
  - `x3 = lambda^2 - x1 - x2`
  - `y3 = lambda * (x1 - x3) - y1`

- The `WeierstrassChip` constrains that these field expressions are computed correctly over the field `C::Fp`.

### 2. `ec_double`

**Assumptions:**

- The point `(x1, y1)` lies on the curve and is not the identity point.

**Circuit statements:**

- The chip takes one input: `(x1, y1)`, and returns `(x3, y3)` where:
  - `lambda = (3 * x1^2 + a) / (2 * y1)`
  - `x3 = lambda^2 - 2 * x1`
  - `y3 = lambda * (x1 - x3) - y1`

- The `WeierstrassChip` constrains that these expressions are computed correctly over the field `C::Fp`. The coefficient `a` is taken from the `CurveConfig`.

### 3. `ec_mul`

**Assumptions:**

- The base point `P` lies in the prime-order subgroup and is not the identity point.
- The 256-bit scalar `k` is odd and less than the subgroup order `n`.
- The subgroup order satisfies `n = 1 (mod 4)`.

**Circuit statements:**

- The chip takes a base point `P` and a little-endian scalar `k`, and returns `k * P`.
- Writing `k = 2B + 1`, the chip uses an MSB-first signed-digit ladder:
  - Initialize `R = P`.
  - For each bit `b` of `B`, update `R = 2 * R + (2 * b - 1) * P`.
- The `EcMulChip` constrains two ladder steps per row, links the accumulator between rows, and checks that the signed digits encode `k`. One multiplication uses 128 rows.
