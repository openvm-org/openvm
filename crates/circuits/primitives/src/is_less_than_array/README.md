# IsLessThanArray

This chip outputs a boolean value `out` that equals 1 if and only if array `x` is lexicographically less than array `y`.

**Assumptions:**
- Input array elements `x[i]` and `y[i]` have a maximum bit length of `max_bits`
- `max_bits` ≤ 29
- `count` is boolean

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
m_i \cdot (m_i - 1) &= 0 & \forall\ i < N &\hspace{2em} \\
s\cdot\left[1 - \sum^{i}_{j=0} m_j \right] \cdot (y_i - x_i) &= 0 & \forall\ i < N &\hspace{2em} \\
m_i \cdot \left[(y_i - x_i) \cdot \texttt{diff\_inv} - 1\right] &= 0 & \forall\ i < N &\hspace{2em} \\
s\cdot\left[1 - \sum^{N-1}_{j=0} m_j\right] \cdot \texttt{out} &= 0 & &\hspace{2em} 
\end{align}
```

Additionally, the chip applies the following constraint:

```math
\begin{align}
\texttt{IsLessThan}\left(\sum^{N-1}_{j=0} m_j \cdot(y_j - x_j),\ \texttt{out},\ s,\ \texttt{lt\_decomp}\right) & &\hspace{2em} 
\end{align}
```

Constraint (1) ensures all $`m_i`$ are boolean (either 0 or 1)

There are two cases to consider:

1. When $`x = y`$ (arrays are identical):
   - Suppose that the constraints are satisfied.
   - Constraint (3) ensures all $`m_i = 0`$ because $m_i \cdot (-1) = 0$ implies $m_i = 0$. 
   - Constraint (4) then forces $`\texttt{out} = 0`$ when $`s \neq 0`$ because $\sum_{j=0}^{N-1} m_j = 0$. 
   - Thus $m$ is all zero and $\texttt{out} = 0$ as desired.
   - Now suppose that $m$ is all zero and $\texttt{out} = 1$.
   - So by $m_i = 0$ for all $i=0,1,\cdots,n-1$ constraint (1) and (3) are satisfied. 
   - Since $x = y$, constraint (2) is also satisfied.
   - Since $\texttt{out} = 0$, constraint (4) is also satisfied.  
   - Hence $m$ is all zero and $\texttt{out} = 0$ if and only if the constraints are satisfied.

2. When $x \neq y$ (arrays are different):
   - Let $k$ be the first index where $x_k \neq y_k$. 
   - Suppose that the constraints are satisfied and $s \not= 0$.
   - For all $i \in \{0,1,\cdots,n-1\}$ such that $x_i = y_i$, constraint (3) satisfied forces $m_i = 0$ because
   
   ```math
   m_i \cdot \bigl((x_i - y_i) \cdot \texttt{diff\_inv}  - 1 \bigr) 
   = m_i \cdot \bigl( 0 \cdot \texttt{diff\_inv} - 1 \bigr) 
   = 0 
   ``` 

   which implies $m_i= 0$. 

   - At index $k$, constraint (2) requires $m_k = 1$ because $y_k - x_k \not= 0$ and, by previous bullet point, all $i < k$ have $m_i = 0$. So $(y_k - x_k) \neq 0$. Thus $1 - \sum_{i=0}^{k} m_i = 1 - m_k = 0$.
   - For all $i > k$ with $x_i \not= y_i$ constraint (2) being satisfied forces $m_i = 0$ by the following reasoning:
      - Let $S = \{ p_0, p_1, \cdots, p_k \}$ be the set of position such that $x_{p_i} \neq y_{p_i}$ for $i \in \{0,1,\cdots,k\}$. So by definition of $k$, $p_0 = k$.
      - For each $i=0,1,\cdots,k$: For each $j < p_i$ and $j \not \in S$ we have $x_j = y_j$, so $m_j = 0$. For each $j < p_i$ and $j \in S$ we have $m_j = 1$ if $j = k$ and $m_j = 0$ otherwise. 
       So
      ```math
      \sum_{j=0}^{p_i} m_j = m_k + m_{p_i} +  \sum_{j=0}^{k-1} m_j + \sum_{j=k+1}^{p_i-1} m_j = 1 + m_{p_i}
      ```
      - Because $x_{p_i} \not= y_{p_i}$ so $1 - \sum_{j=0}^{p_i} m_j = 0$ must hold for constraint (2) to be satisfied at $p_i$. So $1 = \sum_{j = 0}^{p_i} m_j = 1 + m_{p_i}$ which implies $m_{p_i} = 0$.  
   -  Now we have showed that for all positions $i$ that have $x_i = y_i$, $m_i = 0$ must hold. For all positions $i > k$ with $x_i \not= y_i$, $m_i = 0$, and $m_k = 1$. Hence $m_j = 1$ if $j = k$ and $0$ otherwise. 
   - So if the constraints are satisfied then $m$ is $1$ at position $k$ and $0$ everywhere else.

   - Now suppose that $m$ is $1$ at position $k$ and $0$ everywhere else.
   - Then $m$ is a boolean so it satisfies constraint (1).
   - Also $\sum_{j=0}^{i} m_j = 1$ for $i \geq k$ and $y_i - x_i = 0$ for $i < k$. Hence constraint (2) satisfied for all $i$.
   - For all $i$ with $x_i = y_i$ we have $m_i = 0$ so constraint (3) is satisfied. For all $i$ with $x_i \neq y_i$, either $i = k$ or $i > k$. When $i = k$, $(y_i - x_i) \cdot \texttt{diff\_inv} - 1 = (y_i - x_i) \cdot (y_i - x_i)^{-1} - 1 = 0$. When $i > k$, we have $m_i = 0$. So in all three cases, constraint (3) is satisfied.
   - We also know that $\sum_{i=0}^{N-1} m_i = 1$ so constraint (4) is satisfied.

Thus we showed that the constraints are satisfied iff $m$ is as desired. Then we pass $m$ to the $\texttt{IsLessThan}$ AIR which ensures $\texttt{out} = 1$ if and only if $x_k < y_k$. So $\texttt{IsLessThan}$ returns $\texttt{out} = 1$ if and only if $x$ is lexicographically smaller than $y$.

