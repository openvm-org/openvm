# AssertLessThan

This chip verifies if an input value $`x`$ is less than another value $`y`$.

**Assumptions:**
- Input values `x` and `y` have a maximum bit length of `max_bits`
- `max_bits` ≤ 29
- `count` is boolean

This SubAir asserts that `x < y` by range checking that the difference `y - x - 1` has a maximum bit length of `max_bits`.
This is accomplished by taking the limb decomposition of `y - x - 1`, range checking that each limb is valid and constraining that the reconstruction is equal to `y - x - 1`.
Range checking is performed using a lookup table via interactions.

**IO Columns:**
- `x`: First input value to compare
- `y`: Second input value to compare
- `count`: Activation flag $`s`$ (constraints only apply when $`s \neq 0`$)

**Aux Columns:**
- `lower_decomp`: Array of limbs for range checking

**Proof**

Given input values $`x, y \in [0, 2^{\texttt{max\_bits}})`$.

There are two cases to consider:

1. When $`x < y`$:

The difference $`y - x`$ is lies in the range:

```math
0 < y - x < 2^{\texttt{max\_bits}}
```

Since $`x`$ and $`y`$ are field elements, we can rewrite this with an equality at both ends:

```math
\begin{aligned}
   1 &\leq y - x \leq 2^{\texttt{max\_bits}} - 1 \\
   0 &\leq y - x - 1 \leq 2^{\texttt{max\_bits}} - 2
\end{aligned}
```

2. When $`x \geq y`$:

The difference $`y - x`$ lies in the range:

```math
\begin{aligned}
   -(2^{\texttt{max\_bits}} - 1) &\leq y - x \leq 0 \\
   -2^{\texttt{max\_bits}} + 1 &\leq y - x \leq 0 \\
   -2^{\texttt{max\_bits}} &\leq y - x - 1 \leq -1
\end{aligned}
```

Since we're working with field elements over the prime field of order $`p`$, these negative values are represented as their modular equivalents. To make the bounds more intuitive and explicitly positive, we can add $`p`$ to both sides of the inequality:

```math
p - 2^{\texttt{max\_bits}} \leq y - x - 1 \leq p - 1 \mod{p}
```

We can distinguish between these cases using a range check on $`y - x - 1`$ as long as the two sets are non-overlapping. This would be the case when the lower bound $`p - 2^{\texttt{max\_bits}}`$ contains more than `max_bits` bits:

```math
\begin{aligned}
2^{\texttt{max\_bits}} &\leq p - 2^{\texttt{max\_bits}} \\
2^{\texttt{max\_bits+1}} &\leq p \\
\texttt{max\_bits} &\leq \lfloor\log_2(p)\rfloor - 1
\end{aligned}
```

For the babybear field ($`p = 2^{31} - 2^{27} + 1`$), this gives us `max_bits` ≤ 29.
