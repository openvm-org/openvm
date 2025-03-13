# IsLessThanArray

This chip outputs a boolean value `out` that equals 1 if and only if array `x` is lexicographically less than array `y`.

**IO Columns:**
- `x`: Array of input values $`[x_0, x_1, \ldots, x_{n-1}]`$
- `y`: Array of input values $`[y_0, y_1, \ldots, y_{n-1}]`$
- `max_bits`: Maximum bit length of each input value
- `count`: Activation flag $`s`$ (constraints only apply when `count != 0`)
- `out`: Boolean output indicating whether `x < y` lexicographically

**Aux Columns:**
- `diff_marker`: Array where only the first index i with $`x_i \neq y_i`$ is marked with 1, all others are 0
- `diff_inv`: Inverse of the difference $`(y_i - x_i)`$ at the first index where values differ
- `lt_decomp`: Limb decomposition for range checking the difference

The chip operates by finding the first index where the arrays differ, then comparing the values at that position using the `IsLtSubAir`. If the arrays are identical, the output is constrained to be 0.

The comparison is performed by:
1. Identifying the first differing index with the `diff_marker` array
2. Computing the difference value at this position
3. Using the standard `is_less_than` chip to check if this difference indicates `x < y`
4. Range checking the limb decomposition of the difference

**Constraints:**

```math
\begin{align}
m_i \cdot (m_i - 1) &= 0 & \forall\ i < N &\hspace{2em} (1)\\
s\cdot\left[1 - \sum^{i-1}_{k=0} m_k\right] \cdot (y_i - x_i) &= 0 & \forall\ i < N &\hspace{2em} (2)\\
m_i \cdot \left[(y_i - x_i) \cdot \texttt{diff\_inv} - 1\right] &= 0 & \forall\ i < N &\hspace{2em} (3)\\
\sum^{N-1}_{i=0} m_i \cdot \left(\sum^{N-1}_{i=0} m_i - 1\right) &= 0 & &\hspace{2em} (4)\\
s\cdot\left[1 - \sum^{N-1}_{i=0} m_i\right] \cdot \texttt{out} &= 0 & &\hspace{2em} (5)
\end{align}
```

Additionally, the chip applies the following constraint:

```math
\begin{align}
\texttt{IsLessThan}\left(\sum^{N-1}_{i=0} m_i\cdot(y_i - x_i),\ \texttt{out},\ s,\ \texttt{lt\_decomp}\right) & &\hspace{2em} (6)
\end{align}
```

Constraint (1) ensures all $`m_i`$ are boolean (either 0 or 1)

There are two cases to consider:

1. When $`x = y`$ (arrays are identical):
   - Since all $`x_i = y_i`$, constraint (2) is satisfied for all indices.
   - Constraint (3) ensures all $`m_i = 0`$.
   - Constraint (5) then forces $`\texttt{out} = 0`$ when $`s \neq 0`$.

2. When $`x \neq y`$ (arrays differ):
   - Let $`k`$ be the first index where $`x_k \neq y_k`$.
   - For all $`i < k`$, we have $`x_i = y_i`$, so constraint (2) is satisfied when $`m_i = 0`$.
   - At index $`k`$, constraint (2) requires $`m_k = 1`$ since $`y_k - x_k \neq 0`$ and all previous $`m_i = 0`$ where $`i < k`$.
   - For all $`i > k`$, constraint (2) is satisfied when $`\sum^{i}_{j=0} m_j = 1`$. Since $`\sum^{k}_{j=0} m_j = 1`$ and each $`m_i`$ is boolean, this forces all subsequent $`m_i = 0`$.
   - Constraint (6) determines whether $`x_k < y_k`$ and sets $`\texttt{out}`$ accordingly.
