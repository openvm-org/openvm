# IsZero

This chip verifies if an input value $`x`$ is zero.

**IO Columns:**
- `x`: Input value being checked for equality to zero
- `out`: Boolean output indicating whether `x = 0`
- `condition`: Activation flag $s$ (constraints only apply when $s \neq 0$)

**Aux Columns:**
- `inv`: The purported inverse of `x` when `x ≠ 0`

**Constraints:**

```math
\begin{align}
x \cdot \texttt{out} &= 0 & &\hspace{2em} (1)\\
s \cdot \left(\texttt{out} + x \cdot \texttt{inv} - 1\right) &= 0 & &\hspace{2em} (2)
\end{align}
```

Two cases to consider:

1. When input $`x \neq 0`$:
   - Constraint (1) forces $`\texttt{out} = 0`$
   - Constraint (2) requires $`x \cdot \texttt{inv} = 1`$ when $s$ is active
   - Satisfied by setting $`\texttt{inv} = x^{-1}`$, which exists since $`x \neq 0`$

2. When input $`x = 0`$:
   - Constraint (1) is satisfied for any $`\texttt{out}`$ value
   - Constraint (2) forces $`\texttt{out} = 1`$ when $s$ is active
   - The value of $`\texttt{inv}`$ is irrelevant as it's multiplied by zero

When $`s = 0`$, constraint (2) is inactive, and constraint (1) can be satisfied by setting all trace values to zero.

The constraints effectively implement the logic for a zero-check operation: $`\texttt{out} = (x == 0) ? 1 : 0`$ when $s$ is active.
