# Encoder

Efficient encoding of circuit selectors

This SubAir provides a method to encode multiple selectors using fewer columns than the traditional approach. In circuits, we often need selectors to control which logic is active in a given row. The standard approach uses one boolean column per selector, which is inefficient when dealing with many selectors, especially when encoding a large number of selectors (`count`).

**Traditional Approach:**
With n selectors, we typically use n boolean columns, with exactly one active at any time:

```math
\begin{align}
s_i \cdot (s_i - 1) &= 0 & &\hspace{2em} \text{(boolean constraint)}\\
\sum_{i=0}^{n-1} s_i &= 1 & &\hspace{2em} \text{(exactly one active)}
\end{align}
```

This approach requires n columns to represent n selectors, with each selector expression having degree 1.

**Optimized Approach:**
By allowing selector expressions of higher degree ($d$), we can encode the same number of selectors using significantly fewer columns. For a polynomial of degree $d$ with $k$ variables, we can encode $\binom{d+k}{k} - 1$ distinct selectors (the -1 comes from ignoring the case when all columns are zero).

Instead of having one column per selector, we represent each selector as a unique point in a k-dimensional space, where each coordinate takes values between 0 and $d$, with their sum not exceeding $d$. These points form the integer solutions to:

```math
\sum_{i=0}^{k-1} x_i \leq d \quad \text{where} \quad 0 \leq x_i \leq d
```

For example, with $k=2$ variables and degree $d=2$, the solutions are:
```
(0,0)
(1,0)
(2,0)
(0,1)
(1,1)
(0,2)
```

This gives us 6 distinct points to represent 6 selectors using only 2 columns instead of 6.

For each selector point $\mathbf{c}$, we create a polynomial of degree $d$ that equals 1 at that point and 0 at all other points in our solution set. This is achieved through multivariate Lagrange interpolation:

```math
l_{\mathbf{c}}(\mathbf{x}) = \left[\prod_{i=0}^{k-1}\prod_{j=0}^{c_i-1}\frac{x_i - j}{c_i - j}\right] \cdot \left[\prod_{j=0}^{(d - \sum_{i=0}^{k-1} c_i) - 1}\frac{(d - \sum_{i=0}^{k-1} x_i) - j}{(d - \sum_{i=0}^{k-1} c_i) - j}\right]
```

The resulting polynomial has degree $d$ and serves as our selector expression. The first term in the product is zero when any coordinate $x_i$ is less than its target value $c_i$, while the second term is zero when the sum of coordinates exceeds the target sum.

**Constraints:**

The encoder enforces these constraints:

```math
\begin{align}
\prod_{j=0}^{d} (x_i - j) &= 0 & &\hspace{2em} (1)\\
\prod_{j=0}^{d} \left[\left(\sum_{i=0}^{k-1} x_i\right) - j\right] &= 0 & &\hspace{2em} (2)\\
\sum_{i=\texttt{count}}^{N-1} l_{\mathbf{c}_i}(\mathbf{x}) &= 0 & &\hspace{2em} (3)
\end{align}
```

Constraint (1) ensures each coordinate $x_i$ is an integer between 0 and $d$.

Constraint (2) ensures the sum of coordinates is also an integer between 0 and $d$.

Constraint (3) ensures that the point $\mathbf{x}$ must correspond to one of the defined selectors or be the zero point, where $`N`$ is the total number of possible points and `count` is the number of selectors we need to encode. It sums the Lagrange polynomials for all unused points (those beyond the number of flags we actually need) and requires this sum to be zero. This means the current point either represents one of our defined selectors or is the zero point (reserved for invalid/dummy rows).

Together, these constraints guarantee that the point represented by the columns is a valid selector pattern that we've explicitly defined, preventing any undefined or invalid selector patterns from appearing in the trace.
