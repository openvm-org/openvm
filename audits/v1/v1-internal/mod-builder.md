# Modular Arithmetic Chip Builder

Author: [Avaneesh](https://github.com/Avaneesh-axiom)

## 1. Introduction

Scope: [`mod-builder`](https://github.com/openvm-org/openvm/tree/v1.0.0-rc.1/crates/circuits/mod-builder)

Commit: [v1.0.0-rc.1](https://github.com/openvm-org/openvm/releases/tag/v1.0.0-rc.1)

This review is focused on the `mod-builder` framework, which is used to build circuits that constrain arithmetic operations on modular integers.

## 2. Findings

### 2.1 Carry addition could overflow if `range_checker_bits` is too large

**Severity:** Low
**Context:** [save-if-overflow](https://github.com/openvm-org/openvm/blob/ccff2f1ab5170d91243b6ed7bb593fecb4678073/crates/circuits/mod-builder/src/field_variable.rs#L107C1-L110C1)

**Description:** The `FieldVariable::save_if_overflow` method saves the expression if it can produce a carry too large for the range checker to handle.
But if the `FieldVariable::range_checker_bits` field is too large, then `FieldVariable` will allow carries large enough to cause overflow in the `CheckCarryModToZeroSubAir` sub air [here](https://github.com/openvm-org/openvm/blob/ccff2f1ab5170d91243b6ed7bb593fecb4678073/crates/circuits/primitives/src/bigint/check_carry_to_zero.rs#L83C1-L87C1). 

**Proof of concept:** N/A

**Recommendation:** `range_checker_bits` is configured [here](https://github.com/openvm-org/openvm/blob/ccff2f1ab5170d91243b6ed7bb593fecb4678073/crates/circuits/mod-builder/src/builder.rs#L90) in the `ExprBuilder` constructor.
However, we do not have access to the field modulus here, so instead we can add an assertion in `FieldExpr::eval` [here](https://github.com/openvm-org/openvm/blob/ccff2f1ab5170d91243b6ed7bb593fecb4678073/crates/circuits/mod-builder/src/builder.rs#L311C1-L312C1).
Assert that `range_checker_bits + limb_bits < modulus_bits` where `modulus_bits` is the number of bits in the proof system's modulus (not to be confused with the modulus used for arithmetic in mod-builder).

**Resolution:** [fixed in this PR](https://github.com/openvm-org/openvm/pull/1475).
https://github.com/openvm-org/openvm/commit/1037f75cd2054f5fd30a038d21f7abb9a66d5810

What we did instead is make `save_if_overflow` save the expression it has more than `min(range_checker_bits, max_carry_bits)` bits where `max_carry_bits` is an upper bound to ensure overflow does not occur.

## 3. Discussion

### 3.1 Product of variables could be too large to range check if `range_checker_bits` is too small

The `FieldVariable::mul` method calculates the maximum limb value for the product of two `FieldVariable`'s, and it calls `save_if_overflow` to save the expressions if the product's limbs could be too large.
But if the `FieldVariable::range_checker_bits` field is too small, the product's limbs may never be small enough, despite saving both of the expressions.
This was observed when adding support for moduli with more than 32 limbs.

We considered adding an assertion to check that the product of two `FieldVariable`'s in canonical form (i.e. their limbs are all less than `2^limb_bits`) has carries small enough to be range checked with `range_checker_bits` (and similar assertions for `add`, `sub`, `square`, `int_add`, and `int_mul`).
However, this caused problems with the pairing extension.

In particular, multiplying elements in `Fp12` raised the lower bound for `range_checker_bits` from `17` to `21`, which would increase the number of rows in the range checker's air from `2^18` to `2^22`, making it prohibitively large.
The assertion was for the worst-case scenario where each limb value is `2^limb_bits - 1`, but this is unlikely to happen in the pairing extension since the elements of `Fp12` being multiplied are elliptic curve points.
We decided not to add the assertion for a lower bound on `range_checker_bits`.
Note that this doesn't break soundness since the `range_check` function in `bigint` which `mod-builder` uses asserts that the carries are small enough to be range checked.

### 3.2 The scalar in `int_add` and `int_mul` is not asserted to be small

At first glance, it seems to be an issue that the scalar in `FieldVariable::int_add` or `FieldVariable::int_mul` might be large enough to cause an overflow in the variable being added (or multiplied) to.
When you do `x.add_int(c)`, for example, if the expression `x + c` would overflow, then `x` is saved (i.e. a new variable is created).
But if the overflow was caused by `c` being too large, not `x` being too large, then this won't prevent the limbs from overflowing.
The worry is that the circuit produced by `mod-builder` in this case may be unsound because it seems to be allowing limb addition to overflow.

However, we found that in such a case, `mod-builder` will fail to produce a circuit at all.
In short, all the expressions in `mod-builder` are evaluated as overflow int's during eval and if any of the limbs turn out to be large enough that their carries cannot be range checked, a panic occurs.
For this reason, `mod-builder` does not create unsound circuits as a result of overflowing limbs (see details below).

Since `int_add` and `int_mul` are meant to be used to add/multiply small scalars, this behavior is acceptable.
If larger constants need to be used in an expression, the `ExprBuilder::new_const` method may be used.

**Details**: `mod-builder` will panic when it attempts to calculate the carries for the overflow expression resulting from the addition `x + c`.
That is, as part of the eval method of `FieldExpr` here, the expression `x + c` is evaluated as an `OverflowInt<AB::Expr>` and passed into `CheckCarryToModZeroSubAir`'s eval method, which is then passed into `CheckCarryToZeroSubAir`'s eval method here.
This method calculates the max size of the carries of the overflow int and calls `range_check` to range check them to that max size.
The `range_check` function has this assert that checks that the max size of the carries is smaller than the range checker's pre-configured maximum num of bits.
This assertion fails when the carry is too large to range check.
Thus, this assertion would fail when the carry is large enough to overflow (and hence too large to range check).
Note that the carry size of an `OverflowInt<AB::Expr>` is calculated using `OverflowInt::max_overflow_bits` which is a `usize`, so any overflow in the limbs (of type `AB::Expr`) will not be a problem.