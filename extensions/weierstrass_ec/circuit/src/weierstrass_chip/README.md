# Short Weierstrass (SW) Curve Operations

The `ec_add_ne` and `ec_double` instructions are implemented in the `weierstrass_chip` module.

### 1. `ec_add_ne`

**Assumptions:**

- Both points `(x1, y1)` and `(x2, y2)` lie on the curve and are not the identity point.
- `x1` and `x2` are distinct in the coordinate field.

**Circuit statements:**

- The chip takes two inputs: `(x1, y1)` and `(x2, y2)`, and returns `(x3, y3)` where:
  - `lambda = (y2 - y1) / (x2 - x1)`
  - `x3 = lambda^2 - x1 - x2`
  - `y3 = lambda * (x1 - x3) - y1`

- The `EcAddNeChip` constrains that these field expressions are computed correctly over the field `C::Fp`.

### 2. `ec_double`

**Assumptions:**

- The point `(x1, y1)` lies on the curve and is not the identity point.

**Circuit statements:**

- The chip takes one input: `(x1, y1)`, and returns `(x3, y3)` where:
  - `lambda = (3 * x1^2 + a) / (2 * y1)`
  - `x3 = lambda^2 - 2 * x1`
  - `y3 = lambda * (x1 - x3) - y1`

- The `EcDoubleChip` constrains that these expressions are computed correctly over the field `C::Fp`. The coefficient `a` is taken from the `CurveConfig`.
