# Short Weierstrass (SW) Curve Operations

The `sw_ec_add_proj` and `sw_ec_double_proj` instructions are implemented in the `weierstrass_chip` module using projective coordinates.

Points are represented as `(X, Y, Z)` where the affine point is `(X/Z, Y/Z)`. Identity is represented by `Z = 0`.

All formulas use complete addition formulas from [ePrint 2015/1060](https://eprint.iacr.org/2015/1060.pdf) that handle all edge cases without branching.

### 1. `sw_ec_add_proj`

**Input:** Two projective points `(X1, Y1, Z1)` and `(X2, Y2, Z2)` on curve `y^2 = x^3 + ax + b`

**Output:** `(X3, Y3, Z3) = (X1, Y1, Z1) + (X2, Y2, Z2)`

**Formula:** ePrint 2015/1060 Algorithm 1 (general a) or Algorithm 7 (a=0)

The `EcAddExecutor` and its associated `WeierstrassChip` constrain that the projective addition formula is computed correctly over the field `C::Fp`.

### 2. `sw_ec_double_proj`

**Input:** One projective point `(X1, Y1, Z1)` on curve `y^2 = x^3 + ax + b`

**Output:** `(X3, Y3, Z3) = 2 * (X1, Y1, Z1)`

**Formula:** ePrint 2015/1060 Algorithm 3 (general a) or Algorithm 9 (a=0)

The `EcDoubleExecutor` and its associated `WeierstrassChip` constrain that the projective doubling formula is computed correctly over the field `C::Fp`. The coefficients `a` and `b` are taken from the `CurveConfig`.
