# Bitwise Operation Lookup (XOR and Range check)

XOR operation and range checking via lookup table

This chip implements a lookup table approach for XOR operations and range checks for integers of size $`\texttt{NUM\_BITS}`$. The chip provides lookup table functionality for all possible combinations of $`x`$ and $`y`$ values (both in the range $`0..2^{\texttt{NUM\_BITS}}`$), enabling verification of XOR operations and range checks. In the trace, $x$ and $y$ are stored as binary decompositions (`x_bits` and `y_bits` arrays) rather than as full field elements.

The lookup mechanism works through the Bus interface, with other circuits requesting lookups by incrementing multiplicity counters for the operations they need to perform. Each row in the trace corresponds to a specific $(x, y)$ pair.

The chip uses gate-based constraints to generate the trace columns instead of a preprocessed trace. The trace enumerates all valid $(x, y)$ pairs in order: row $n$ corresponds to $(x, y)$ where $x = \lfloor n / 2^{\texttt{NUM\_BITS}} \rfloor$ and $y = n \bmod 2^{\texttt{NUM\_BITS}}$. The enumeration order is: $(0, 0)$, $(0, 1)$, ..., $(0, 2^{\texttt{NUM\_BITS}}-1)$, $(1, 0)$, $(1, 1)$, ..., up to $(2^{\texttt{NUM\_BITS}}-1, 2^{\texttt{NUM\_BITS}}-1)$.

**Columns:**
- `x_bits[0..NUM_BITS-1]`: Binary decomposition of $x$ (where `x_bits[0]` is the least significant bit)
- `y_bits[0..NUM_BITS-1]`: Binary decomposition of $y$ (where `y_bits[0]` is the least significant bit)
- `mult_range`: Multiplicity column tracking the number of range check operations requested for each $(x, y)$ pair
- `mult_xor`: Multiplicity column tracking the number of XOR operations requested for each $(x, y)$ pair

The constraints enforce the enumeration pattern by:
1. Ensuring each bit is binary (0 or 1) using `assert_bool` constraints
2. Reconstructing $x$ and $y$ from their binary decompositions: $x = \sum_{i=0}^{\texttt{NUM\_BITS}-1} \texttt{x\_bits}[i] \cdot 2^i$
3. Computing $z_{\texttt{xor}} = x \oplus y$ algebraically from bits: $z_{\texttt{xor}} = \sum_{i=0}^{\texttt{NUM\_BITS}-1} (\texttt{x\_bits}[i] + \texttt{y\_bits}[i] - 2 \cdot \texttt{x\_bits}[i] \cdot \texttt{y\_bits}[i]) \cdot 2^i$
4. Constraining that the combined index $(x \cdot 2^{\texttt{NUM\_BITS}} + y)$ increments by 1 each row using transition constraints
5. Enforcing boundary conditions: first row has index 0, last row has index $2^{2 \cdot \texttt{NUM\_BITS}} - 1$
