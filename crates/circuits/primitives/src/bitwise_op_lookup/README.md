# Bitwise Operation Lookup (XOR and Range check)

XOR operation and range checking via lookup table

This chip implements a lookup table approach for XOR operations and range checks for integers of size $`\texttt{NUM\_BITS}`$. The lookup table contains all possible combinations of $`x`$ and $`y`$ values (both in the range $`0..2^{\texttt{NUM\_BITS}}`$), along with their XOR result.

The lookup mechanism works through the Bus interface, with other circuits requesting lookups by incrementing multiplicity counters for the operations they need to perform. Each row in the lookup table corresponds to a specific $(x, y)$ pair.

**Preprocessed Columns:**
- `x`: Column containing the first input value ($0$ to $`2^{\texttt{NUM\_BITS}}-1`$)
- `y`: Column containing the second input value ($0$ to $`2^{\texttt{NUM\_BITS}}-1`$)
- `z_xor`: Column containing the XOR result of x and y ($x \oplus y$)

**IO Columns:**
- `mult_range`: Multiplicity column tracking the number of range check operations requested for each $(x, y)$ pair
- `mult_xor`: Multiplicity column tracking the number of XOR operations requested for each $(x, y)$ pair
