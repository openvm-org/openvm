# IsLessThan

This chip outputs a boolean value `out` that equals 1 if and only if `x` < `y`.

**Assumptions:**
- Input values `x` and `y` have a maximum bit length of `max_bits`
- `max_bits` ≤ 29

This SubAir outputs a boolean value `out` that equals 1 if and only if `x < y`. It operates similarly to `assert_less_than` but includes an additional output column.
The verification is accomplished by computing a shifted difference value, taking its limb decomposition, and range checking the limbs while constraining their reconstruction.

**IO Columns:**
- `x`: First input value to compare
- `y`: Second input value to compare
- `out`: Boolean output indicating whether `x < y`
- `count`: Activation flag $`s`$ (constraints only apply when $`s \neq 0`$)

**Aux Columns:**
- `lower_decomp`: Array of limbs for range checking

**Proof**

Given input values $`x, y \in [0, 2^{\texttt{max\_bits}})`$.

The SubAir computes $`y - x - 1`$, decomposes the lower `max_bits` bits of this difference into limbs and range checks that each limb is valid.

There are two cases to consider:

1. When $`x < y`$:

```math
\begin{aligned}
   0 &< y - x < 2^{\texttt{max\_bits}} \\
   1 &\leq y - x \leq 2^{\texttt{max\_bits}} - 1 \\
   0 &\leq y - x - 1 \leq 2^{\texttt{max\_bits}} - 2\\
   2^{\texttt{max\_bits}} &\leq y - x - 1 + 2^{\texttt{max\_bits}} \leq 2^{\texttt{max\_bits}+1} - 2
\end{aligned}
```

From the lower `max_bits` limb decomposition of $`y - x - 1`$, we constrain that the reconstructed value is equal to $`y - x - 1 - 2^{\texttt{max\_bits}}`$.

2. When $`x \geq y`$:

```math
\begin{aligned}
   -(2^{\texttt{max\_bits}} - 1) &\leq y - x \leq 0 \\
   -2^{\texttt{max\_bits}} + 1 &\leq y - x \leq 0 \\
   -2^{\texttt{max\_bits}} &\leq y - x - 1 \leq -1 \\
   0 &\leq y - x - 1 + 2^{\texttt{max\_bits}} \leq 2^{\texttt{max\_bits}} - 1
\end{aligned}
```

From the lower `max_bits` limb decomposition of $`y - x - 1`$, we constrain that the reconstructed value is equal to $`y - x - 1`$.

We combine constraints for both cases by constraining that the reconstructed value equals $`y - x - 1 - \texttt{out} \cdot 2^{\texttt{max\_bits}}`$.

**Constraints:**

The main constraint enforces that the reconstructed value from the limbs (`lower`) plus $`\texttt{out} \cdot 2^{\texttt{max\_bits}}`$ equals the intermediate value $`(y - x - 1 + 2^{\texttt{max\_bits}})`$:

```math
\begin{align}
s \cdot \left(\texttt{lower} + \texttt{out} \cdot 2^{\texttt{max\_bits}} - (y - x - 1 + 2^{\texttt{max\_bits}})\right) &= 0 & &\hspace{2em} (1)\\
\texttt{out} \cdot (\texttt{out} - 1) &= 0 & &\hspace{2em} (2)
\end{align}
```

The second constraint ensures that `out` is a boolean value (0 or 1).

Additionally, the SubAir interacts with a range checker to verify that each limb in `lower_decomp` has the appropriate number of bits.
