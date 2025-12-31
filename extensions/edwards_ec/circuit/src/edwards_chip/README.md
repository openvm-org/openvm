# Twisted Edwards (TE) curve operations

The `te_add` instruction is implemented in the `edwards_chip` module.

### 1. `te_add`

**Assumptions:**

- Both points `(x1, y1)` and `(x2, y2)` lie on the curve.

**Circuit statements:**

- The chip takes two inputs: `(x1, y1)` and `(x2, y2)`, and returns `(x3, y3)` where:
  - `x3 = (x1 * y2 + x2 * y1) / (1 + d * x1 * x2 * y1 * y2)`
  - `y3 = (y1 * y2 - a * x1 * x2) / (1 - d * x1 * x2 * y1 * y2)`

- The `TeAddChip` constrains that these field expressions are computed correctly over the field `C::Fp`. The coefficients `a` and `d` are taken from the `CurveConfig`.
