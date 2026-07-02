# Circuit Primitives

Author: [shuklaayush](https://github.com/shuklaayush)

## 1. Introduction

Scope: [`circuit-primitives`](https://github.com/openvm-org/openvm/blob/main/crates/circuits/primitives/)

Commit: [efdcdd76320729e2b323835da5a368d5780e1e4d](https://github.com/openvm-org/openvm/commit/efdcdd76320729e2b323835da5a368d5780e1e4d)

This review focuses on core SubAirs that are reused across the codebase, reviewing their stark-backend and circuit interface usage, while documenting and validating their core assumptions and claims.

## 2. Findings

### 2.1 Incorrect debug assertion in `decompose` method

**Severity:** Medium

**Context:** https://github.com/openvm-org/openvm/blob/efdcdd76320729e2b323835da5a368d5780e1e4d/crates/circuits/primitives/src/var_range/mod.rs#L165-L169

**Description:** In the `decompose` method of the `VariableRangeCheckerBus`, there is an incorrect assertion that checks if there are enough limbs. The current check `limbs.len() <= bits.div_ceil(self.range_max_bits())` should actually be `limbs.len() >= bits.div_ceil(self.range_max_bits())`.

**Recommendation:** Modify the assertion to check that `limbs.len() >= bits.div_ceil(self.range_max_bits())` to ensure there are enough limbs to hold the decomposed value.

**Resolution:** https://github.com/openvm-org/openvm/pull/1454
https://github.com/openvm-org/openvm/commit/cb966bb09cef9d199b29f5081fd652517e8d6937

### 2.2 Integer overflows in `OverflowInt` methods

**Severity:** Low

**Context:** https://github.com/openvm-org/openvm/blob/efdcdd76320729e2b323835da5a368d5780e1e4d/crates/circuits/primitives/src/bigint/mod.rs#L89

**Description:** Most of the `isize` and `usize` operations in this file are unchecked and can overflow/underflow. Note that in Rust, integer overflow causes a panic in debug mode but usually wraps around in release mode (see [overflow-checks](https://doc.rust-lang.org/cargo/reference/profiles.html#overflow-checks) and [book](https://doc.rust-lang.org/stable/book/ch03-02-data-types.html#integer-overflow)).

**Recommendation:** We should use checked/strict arithmetic operations to prevent integer overflows and underflows.

**Resolution:** None

### 2.3 Redundant constraint in `is_less_than_array`

**Severity:** Low

**Context:** https://github.com/openvm-org/openvm/blob/efdcdd76320729e2b323835da5a368d5780e1e4d/crates/circuits/primitives/src/is_less_than_array/mod.rs#L135

**Description:** This constraint in the `is_less_than_array` implementation is redundant since the other constraints already ensure that at most only one `m_i` can be 1

**Recommendation:** Consider removing this constraint

**Resolution:** None

### 2.4 Redundant `max_overflow_bits` parameter in `OverflowInt`

**Severity:** Informational

**Context:** https://github.com/openvm-org/openvm/blob/efdcdd76320729e2b323835da5a368d5780e1e4d/crates/circuits/primitives/src/bigint/mod.rs#L94-L96

**Description:** The `OverflowInt` struct tracks both `limb_max_abs` and `max_overflow_bits`, but `max_overflow_bits` could be derived from `limb_max_abs` using `max_overflow_bits = ceil(log2(limb_max_abs))`.

**Recommendation:** Consider making `max_overflow_bits` a computed property based on `limb_max_abs` rather than storing it separately.

**Resolution:** None

### 2.5 Unnecessary creation of `VariableRangeCheckerBus` in range check utility

**Severity:** Informational

**Context:** https://github.com/openvm-org/openvm/blob/efdcdd76320729e2b323835da5a368d5780e1e4d/crates/circuits/primitives/src/bigint/utils.rs#L24-L33

**Description:** In the `range_check` utility function, a new `VariableRangeCheckerBus` is created for every call. This is inefficient as the bus could potentially be reused across multiple range check operations.

**Recommendation:** Consider refactoring to allow reusing the same `VariableRangeCheckerBus` instance across multiple range check calls when appropriate like in other chips.

**Resolution:** None

### 2.6 Unused `is_equal_array/trace.rs` file

**Severity:** Informational

**Context:** https://github.com/openvm-org/openvm/blob/efdcdd76320729e2b323835da5a368d5780e1e4d/crates/circuits/primitives/src/is_equal_array/trace.rs

**Description:** The file `is_equal_array/trace.rs` exists in the repository but is not used anywhere.

**Recommendation:** Remove the file.

**Resolution:** None

### 2.7 Missing boolean assumption for `count` in `assert_less_than` and `is_less_than`

**Severity:** Low

**Context:** https://github.com/openvm-org/openvm/blob/efdcdd76320729e2b323835da5a368d5780e1e4d/crates/circuits/primitives/src/assert_less_than/mod.rs#L25-L38

**Description:** `count` is assumed to be boolean in `assert_less_than` and `is_less_than` but this is not mentioned anywhere.

**Recommendation:** Explicitly mention that `count` is assumed to be boolean and verify that this indeed is the case wherever this subair is used.

**Resolution:** https://github.com/openvm-org/openvm/pull/1453
https://github.com/openvm-org/openvm/commit/8974f018489e8512d35734e0f377551af66fa334

## 3. Discussion

This section evaluates the design and implementation of circuit primitives, examining both standalone AIRs and SubAIRs.

### 3.1 Standalone Airs

#### 3.1.1 [range](https://github.com/openvm-org/openvm/tree/main/crates/circuits/primitives/src/range)

Range checking

This chip implements range checking using a lookup table approach. The lookup table contains:
1. All possible values within the range $`0..\texttt{range\_max}`$
2. A column tracking the multiplicity of each value (how many times each value needs to be range checked)

Range check constraints are enforced through interactions between this range chip and other chips that need to perform range checks.

**Preprocessed Columns:**
- `counter`: Column containing all possible values within the range $`0..\texttt{range\_max}`$

**IO Columns:**
- `mult`: Multiplicity column that tracks the number of range checks to perform for each element

**Note:** This implementation can also be used to efficiently range check values up to $`2^{\texttt{max\_bits}}`$ (where $`2^{\texttt{max\_bits}} < \texttt{range\_max}`$) by checking both the original value and a shifted value. This works by verifying that both $`x`$ and $`x + (\texttt{range\_max} - 2^{\texttt{max\_bits}})`$ are within the valid range of $`0..\texttt{range\_max}`$. This approach only works when $`2 * \texttt{range\_max}`$ is less than the field modulus to avoid wrap-around issues.

#### 3.1.2 [range_gate](https://github.com/openvm-org/openvm/tree/main/crates/circuits/primitives/src/range_gate)

Range checking without preprocessed columns

This chip implements range checking using a lookup table approach, but with dynamically generated counter values in the trace rather than using preprocessed columns.

**Columns:**
- `counter`: Dynamically generated column containing sequential values from 0 to `range_max-1`
- `mult`: Multiplicity column tracking the number of range checks requested for each value

**Constraints:**

```math
\begin{align}
\texttt{counter}_0 &= 0 & &\hspace{2em} (1)\\
\texttt{counter}_{i+1} - \texttt{counter}_i &= 1 & \forall\ 0 \leq i < H-1 &\hspace{2em} (2)\\
\texttt{counter}_{H-1} &= \texttt{range\_max} - 1 & &\hspace{2em} (3)
\end{align}
```

Constraint (1) ensures the counter starts at 0 on the first row.

Constraint (2) enforces that each subsequent row increments the counter by exactly 1.

Constraint (3) verifies that the last row contains the value `range_max-1`, ensuring the trace has the correct height.

The trace is generated by accumulating the count of range checks requested for each value in the `mult` column, with each row's `counter` value representing the number being checked.

#### 3.1.3 [range_tuple](https://github.com/openvm-org/openvm/tree/main/crates/circuits/primitives/src/range_tuple)

Tuple-based range checking for multiple values simultaneously

This chip efficiently range checks tuples of values using a single interaction when the product of their ranges is relatively small (less than ~2^20). For example, when checking pairs `(x, y)` against their respective bit limits, this approach is more efficient than performing separate range checks.

**Preprocessed Columns:**
- `tuple`: Column containing all possible tuple combinations within the specified ranges

**IO Columns:**
- `mult`: Multiplicity column tracking the number of range checks requested for each tuple

The implementation creates a preprocessed table with all possible value combinations within the specified ranges. The `sizes` parameter in `RangeTupleCheckerBus` defines the maximum value for each dimension.

For a 2-dimensional tuple with `sizes = [3, 2]`, the preprocessed trace contains these 6 combinations in lexicographic order:
```
(0,0)
(1,0)
(2,0)
(0,1)
(1,1)
(2,1)
```

During circuit execution, each row corresponds to a specific tuple from the preprocessed trace, with the `mult` column recording how many times that tuple was requested for range checking.

#### 3.1.4 [var_range](https://github.com/openvm-org/openvm/tree/main/crates/circuits/primitives/src/var_range)

Variable range checking

This chip implements a lookup table for range checking values with variable bit sizes. Unlike standard range checking that works with a fixed bit size, this chip verifies that a value `x` has `b` bits, where `b` can be any integer from 0 to `range_max_bits`.

Conceptually, this works like `range_max_bits` different lookup tables stacked together:
- One table for 1-bit values
- One table for 2-bit values
- And so on up to `range_max_bits`-bit values

With a selector column indicating which bit-size to check against.

For example, with `range_max_bits = 3`, the lookup table contains:
- All 1-bit values: 0, 1
- All 2-bit values: 0, 1, 2, 3
- All 3-bit values: 0, 1, 2, 3, 4, 5, 6, 7

By convention, the value 0 is defined to have 0 bits.

**Preprocessed Columns:**
- `value`: The value being range checked
- `max_bits`: The maximum number of bits for this value

**IO Columns:**
- `mult`: Multiplicity column tracking how many range checks are requested for each (value, max_bits) pair

#### 3.1.5 [xor](https://github.com/openvm-org/openvm/tree/main/crates/circuits/primitives/src/xor)

XOR operation verification via lookup table

This chip implements a lookup table approach for XOR operations on integers with a maximum of `\texttt{M}` bits. The lookup table contains all possible combinations of `x` and `y` values (both in the range $`0..2^{\texttt{M}}`$), along with their XOR result.

The core functionality works through the `XorBus` interface, with other circuits requesting XOR operations by incrementing the appropriate multiplicity counter for each (x, y) pair. Each row in the lookup table corresponds to a specific (x, y) pair and tracks its usage count.

**Preprocessed Columns:**
- `x`: Column containing the first input value ($0$ to $`2^\texttt{M}-1`$)
- `y`: Column containing the second input value ($0$ to $`2^\texttt{M}-1`$)
- `z`: Column containing the XOR result of x and y ($x \oplus y$)

**IO Columns:**
- `mult`: Multiplicity column tracking the number of XOR operations requested for each $(x, y)$ pair

#### 3.1.6 [bitwise_op_lookup](https://github.com/openvm-org/openvm/tree/main/crates/circuits/primitives/src/bitwise_op_lookup)

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

### 3.2 SubAirs

#### 3.2.1 [assert_less_than](https://github.com/openvm-org/openvm/tree/main/crates/circuits/primitives/src/assert_less_than)

Less than assertion checking for `x` < `y`

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

#### 3.2.2 [bigint](https://github.com/openvm-org/openvm/tree/main/crates/circuits/primitives/src/bigint)

Big integer operations.

We define the canonical representation (or proper BigInt representation) of an integer $`x`$ as an array of limbs $`[a_0, a_1, ..., a_{n-1}]`$ where each $`0 \leq a_i < 2^{\texttt{limb\_bits}}`$ such that $`x = \sum_{i=0}^{n-1} a_i \cdot 2^{i \cdot \texttt{limb\_bits}}`$. This is essentially a base-$`2^{\texttt{limb\_bits}}`$ little-endian representation.

The signed overflow representation is defined as $`[a_0, a_1, ..., a_{n-1}]`$ where each limb $`a_i`$ is signed and allowed to have an absolute value potentially larger than $`2^{\texttt{limb\_bits}}`$, specifically $`-2^{\texttt{overflow\_bits}} \leq a_i < 2^{\texttt{overflow\_bits}}`$ where $`\texttt{overflow\_bits} \geq \texttt{limb\_bits}`$. The value of the integer is still calculated as $`x = \sum_{i=0}^{n-1} a_i \cdot 2^{i \cdot \texttt{limb\_bits}}`$.

This overflow representation is useful for optimizing constraints in intermediate calculations, as it allows us to store the results of operations without immediately performing carry propagation. Each limb can be thought of as the intermediate result of arithmetic operations without normalizing to the canonical form.

To ensure each overflow limb has a unique representation in the field, we need to ensure that the negative values don't overlap with the positive values:

```math
\begin{align}
p - 2^{\texttt{overflow\_bits}} &> 2^{\texttt{overflow\_bits}} - 1 \\
p - 2^{\texttt{overflow\_bits}} &\geq 2^{\texttt{overflow\_bits}} \\
2^{\texttt{overflow\_bits+1}} &\leq p \\
\texttt{overflow\_bits} &\leq \lfloor\log_2(p)\rfloor - 1
\end{align}
```

When performing operations with numbers in this overflow representation, we need to ensure the absolute value of each limb remains within appropriate bounds to avoid field modulus issues.

The canonical representation can be calculated by reducing each limb modulo $`2^{\texttt{limb\_bits}}`$ and propagating the carry to the next limb.

##### 3.2.3 [check_carry_to_zero](https://github.com/openvm-org/openvm/blob/main/crates/circuits/primitives/src/bigint/check_carry_to_zero.rs)

This SubAir constrains that a given overflow limb representation of an integer is zero. This is done by taking the carries as hint, range-checking that they are valid, verifying that they are correct and ensuring that the final carry is zero.

**IO Columns:**
- `is_valid`: Boolean selector $`s`$ indicating whether the row is a real (non-padding) row
- `limbs`: Array of overflow limbs representing a big integer $`[a_0, a_1, \ldots, a_{n-1}]`$

**Aux Columns:**
- `carries`: Array of carries for converting overflow limbs to canonical representation $`[c_0, c_1, \ldots, c_{n-1}]`$. Carries are allowed to be negative and should be within the range $`[-2^{\texttt{overflow\_bits} - \texttt{limb\_bits}} \leq c_i < 2^{\texttt{overflow\_bits} - \texttt{limb\_bits}})`$.

In addition to the `limbs`, we also keep a track of the maximum value that can be represented by the overflow representation. If the given input is in canonical representation then this value usually starts at $`2^{\texttt{limb\_bits}} - 1`$ and expands as operations are performed on the overflow representation. This is similar to how interval arithmetic is used to track the bounds of a value. The bound is useful to find the least number of overflow bits needed to represent the integer and this can be used to perform smallest range checks on the carries.

**Constraints:**

To range check the carries within the range $`-2^{\texttt{overflow\_bits} - \texttt{limb\_bits}} \leq c_i < 2^{\texttt{overflow\_bits} - \texttt{limb\_bits}}`$, we add $`-2^{\texttt{overflow\_bits} - \texttt{limb\_bits}}`$ to both sides of the range to make the range check from 0 to some value, only when selector is on.

```math
0 \leq c_i + 2^{\texttt{overflow\_bits} - \texttt{limb\_bits}} < 2^{\texttt{overflow\_bits} - \texttt{limb\_bits} + 1}
```

We check the carries are calculated correctly and the integer represented by the limbs is zero.

```math
\begin{align}
a_0 &= c_0 \cdot 2^{\texttt{limb\_bits}} & &\hspace{2em} (1)\\
a_i + c_{i-1} &= c_i \cdot 2^{\texttt{limb\_bits}} & \forall\ 0 < i < N &\hspace{2em} (2)\\
c_{N-1} &= 0 & &\hspace{2em} (3)
\end{align}
```

What this is demonstrating is that there exists an array of carries $`[c_0, c_1, \ldots, c_{n-1}]`$ which represent the carries for converting overflow limbs to canonical representation such that the canonical representation of the integer is zero.

##### 3.2.3 [check_carry_mod_to_zero](https://github.com/openvm-org/openvm/blob/main/crates/circuits/primitives/src/bigint/check_carry_to_zero.rs)

This SubAir constrains that a given overflow limb representation of an integer is $`x - q * m = 0`$ i.e. $`x = 0\mod m`$.

**IO Columns:**
- `is_valid`: Boolean selector $`s`$ indicating whether the row is a real (non-padding) row
- `limbs`: Array of overflow limbs representing a big integer $`[a_0, a_1, \ldots, a_{n-1}]`$

**Aux Columns:**
- `quotient`: Array of quotient limbs representing a big integer $`[q_0, q_1, \ldots, q_{n-1}]`$ where
- `carries`: Array of carries for converting overflow limbs to canonical representation $`[c_0, c_1, \ldots, c_{n-1}]`$. Carries are allowed to be negative and should be within the range $`[-2^{\texttt{overflow\_bits} - \texttt{limb\_bits}} \leq c_i < 2^{\texttt{overflow\_bits} - \texttt{limb\_bits}})`$.

**Constraints:**

We range check the quotient to be a valid signed `limb_bit` representation i.e. $`-2^{\texttt{limb\_bits}} \leq q_i < 2^{\texttt{limb\_bits}}`$. To range check within this range, we add $`2^{\texttt{limb\_bits}}`$ to both sides to make the range check from 0 to some value, only when selector is on. Quotient can be negative.

```math
0 \leq q_i + 2^{\texttt{limb\_bits}} < 2^{\texttt{limb\_bits}+1}
```

Finally, we calculate the remainder limbs $`[r_0, r_1, \ldots, r_{n-1}]`$ where $`r_i = a_i - q_i * m_i`$ and constrain it to 0.

```math
\begin{align}
\texttt{CheckCarryToZero}(a - q * m) & &\hspace{2em} (1)
\end{align}
```

What this is demonstrating is that there exists a quotient $`q`$ such that $`a - q * m = 0`$.

#### 3.2.3 [encoder](https://github.com/openvm-org/openvm/tree/main/crates/circuits/primitives/src/encoder)

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

#### 3.2.4 [is_equal](https://github.com/openvm-org/openvm/tree/main/crates/circuits/primitives/src/is_equal)
Equality comparison

This SubAir outputs a boolean value `out` that equals 1 if and only if `x` equals `y`.

**IO Columns:**
- `x`: First input value to compare
- `y`: Second input value to compare
- `out`: Boolean output indicating whether `x = y`
- `condition`: Activation flag $s$ (constraints only apply when $s \neq 0$)

**Aux Columns:**
- `inv`: The purported inverse of `x - y` when `x ≠ y`

The SubAir leverages the `IsZeroSubAir` by checking if the difference `x - y` is zero.

**Constraints:**

The IsEqualSubAir applies the following constraint:

```math
\begin{align}
\texttt{IsZero}(x - y,\ \texttt{out},\ s,\ \texttt{inv}) & &\hspace{2em} (1)
\end{align}
```

Where the IsZeroSubAir implements constraints to ensure that `out` is 1 if and only if `x - y` is zero (which happens exactly when `x = y`).

#### 3.2.5 [is_equal_array](https://github.com/openvm-org/openvm/tree/main/crates/circuits/primitives/src/is_equal_array)
Array equality comparison

This SubAir outputs a boolean value `out` that equals 1 if and only if arrays `x` and `y` are equal.

**IO Columns:**
- `x`: Array of input values `[x_0, x_1, ..., x_{n-1}]`
- `y`: Array of input values `[y_0, y_1, ..., y_{n-1}]`
- `out`: Boolean output indicating whether `x = y`
- `condition`: Activation flag `s` (constraints only apply when `s != 0`)

**Aux Columns:**
- `diff_inv_marker`: Array where only the first index i with `x[i] != y[i]` contains the inverse of `x[i] - y[i]`, all others are 0

**Constraints:**

```math
\begin{align}
\texttt{out} \cdot (x_i - y_i) &= 0 & \forall\ i < N &\hspace{2em} (1)\\
s \cdot \left(\texttt{out} + \sum_{i=0}^{N-1} (x_i - y_i) \cdot \texttt{diff\_inv\_marker}_i - 1\right) &= 0 & &\hspace{2em} (2)\\
\texttt{out} \cdot (\texttt{out} - 1) &= 0 & &\hspace{2em} (3)
\end{align}
```

Two cases to consider:

1. When arrays are equal (`x = y`):
   - All differences `x_i - y_i` are zero, so constraint (1) is satisfied for any `out` value
   - In constraint (2), the sum term becomes zero, forcing `out = 1` when $s$ is active
   - Constraint (3) ensures `out` is boolean

2. When arrays are not equal:
   - Let `k` be the first index where `x_k ≠ y_k`
   - Constraint (1) forces `out = 0` since the product must be zero
   - For constraint (2), the prover sets `diff_inv_marker[k] = (x_k - y_k)^{-1}` and all other entries to zero
   - This makes the sum term equal to 1, satisfying constraint (2) when `out = 0`
   - Constraint (3) is satisfied since `out = 0`

The SubAir doesn't explicitly enforce that only the first differing index has a non-zero `diff_inv_marker` value, or that it contains the exact inverse. It only requires that the weighted sum of differences equals 1 when arrays differ. However, the trace generation sets these values correctly for efficiency.

#### 3.2.6 [is_less_than](https://github.com/openvm-org/openvm/blob/main/crates/circuits/primitives/src/is_less_than)
Less than comparison for outputting a boolean indicating `x` < `y`

**Assumptions:**
- Input values `x` and `y` have a maximum bit length of `max_bits`
- `max_bits` ≤ 29
- `count` is boolean

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

#### 3.2.7 [is_less_than_array](https://github.com/openvm-org/openvm/tree/main/crates/circuits/primitives/src/is_less_than_array)

Array less than comparison

This SubAir outputs a boolean value `out` that equals 1 if and only if array `x` is lexicographically less than array `y`.

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

The SubAir operates by finding the first index where the arrays differ, then comparing the values at that position using the `IsLtSubAir`. If the arrays are identical, the output is constrained to be 0.

The comparison is performed by:
1. Identifying the first differing index with the `diff_marker` array
2. Computing the difference value at this position
3. Using the standard `is_less_than` SubAir to check if this difference indicates `x < y`
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

Additionally, the SubAir applies the following constraint:

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

#### 3.2.8 [is_zero](https://github.com/openvm-org/openvm/tree/main/crates/circuits/primitives/src/is_zero)

This SubAir verifies if an input value $`x`$ is zero.

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
