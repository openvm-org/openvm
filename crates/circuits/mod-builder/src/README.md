# Mod Builder

Mod Builder is a framework used to build circuits that constrain arithmetic operations on modular integers.
See the [usage](#usage) section to get started.

Note: Mod Builder assumes the proof system modulus is 31 bits.

## Overview

The goal of Mod Builder is:
- given $n$ input variables $x_1, ..., x_n \in \mathbb{Z}_p$, and
- given $m$ rational functions $f_1, ..., f_m$ from $\mathbb{Z}_p^n$ to $\mathbb{Z}_p$ (i.e. only $+, -, *, /$ are allowed),

to produce a circuit with variables $x_1, ..., x_n, y_1, ..., y_m$ that constrains $y_i \equiv f_i(x_1, ..., x_n) \pmod{p}$ for all $i = 1, ..., m$.

Note: actually, $f_i$ is allowed to use the result of $f_j$ for $j < i$, but for notational simplicity we ignore this.

We use the following trick to constrain modular arithmetic.
Suppose we want to constrain $z = x + y \pmod{p}$.
We can represent this as the constraint $x + y - z + q p = 0$ in the integers, where $q$ is a new witness variable.
This idea works for addition, subtraction, and multiplication.
To support division, we use the fact that $x y^{-1} = z \pmod{p}$ is equivalent to $x = z y \pmod{p}$.

To support moduli larger than 31 bits, we represent modular integers as an array of limbs, where each limb has `limb_bits` bits.

## An Optimization

Mod builder works by transforming the rational functions into a set of constraint expressions, using the trick described above.
Then we constrain each expression equal to zero by doing the big integer arithmetic between the multi-limb variables.

We make the optimization that while evaluating each constraint, we allow the limbs of the variables to have more than `limb_bits` bits.
After all operations are performed, we reduce the limbs back to `limb_bits` size (i.e. we canonicalize the variable) and constrain it equal to zero.

For example, if we want to constrain $x + y + qp - z = 0$ then we symbolically evaluate the expression to obtain a list of `AB::Expr` of length `NUM_LIMBS`, where each limb is at most `limb_bits` bits.
Then we canonicalize the variable by iterating from the least significant limb to the most significant and carrying the overflow to the next limb.
Finally, we constrain that the last carry is zero.
Note that we need to make each carry a new witness variable due to the limit on the maximum constraint degree.

See the [bigint module ](https://github.com/openvm-org/openvm/blob/main/crates/circuits/primitives/src/bigint/README.md) for details.

Note: each constraint must be small enough that the limbs of $c$ are not too large, otherwise they will overflow the modulus.
We ensure that this is never the case by breaking large constraints into smaller ones.
See the [Saving Variables](#saving-variables) section for details.

## Constraints vs Computes

Mod Builder maintains two types of expressions: constraints and computes.
A constraint is an expression that will be constrained to zero **modulo $p$** by the circuit.
For example, to constrain $z = x + y \pmod{p}$, we build the constraint $x + y - z$.
(since this constraint is modulo $p$, it will actually be constrained as $x + y - z - qp = 0$ in the circuit)

A "compute" is what we call an expression that describes how to compute a variable from other variables, and it is used in trace generation.
In our example, the compute is $x + y$ and it computes $z$.
Every variable besides the input variables has exactly one compute expression associated with it, and every compute expression is associated with exactly one constraint.

The distinction between constraints and computes is most clear when we constrain division.
For example, to constrain $z = x y^{-1} \pmod{p}$, we build the constraint $z y - x$ and the compute $x y^{-1}$.
Note that computes are allowed to have division since they are only used in trace generation, while constraints are not.

A circuit built by Mod Builder can be mainly described by its constraints and computes, and much of Mod Builder's purpose is to easily create constraints and computes representing arbitrary modular arithmetic expressions.

## Saving Variables
The `FieldVariable` builds constraints and computes.
It stores an expression tree, and its arithmetic operators are overloaded to build the expression tree.
For example, if `x` and `y` are `FieldVariable`s, then `let z = x + y` will create a new `FieldVariable` that stores the expression $x + y$.

The `FieldVariable` struct also decides when to create a new intermediate variable.
It maintains the maximum possible size of the limbs of its expression, and when it detects that the limbs could be too big (specifically, big enough that the constraints described [earlier](#an-optimization) will overflow the modulus), it creates a new intermediate variable and adds the appropriate constraint and compute.

For example, suppose the constraint $xy + a - z - qp = 0$ could overflow.
If we do `let z = x * y + a` then `z` will be a `FieldVariable` that stores the expression $z$ (not $xy + a$) and there will be a new constraint $xy + a - z$ and a new compute $xy + a$ associated with `z`.
Note that `z` is a new witness variable in the circuit.
We call this operation a "save" and we say that we "saved" the expression $xy + a$ into a new variable $z$.

We can continue to use `z` as usual, for instance `let w = z + b` will result in `w` storing the constraint $z + b$.
When we are done, we call `w.save_output()` to save $w$.
This will add the constraint $z + b - w$ and the compute $z + b$.
It will also mark $w$ as an output variable, which means it will be written to memory by the `FieldExpressionCoreChip`.

Note that the variables in the expression tree don't actually have names, they are just numbered starting from 0.

### Where are the constraints and computes stored?

The `ExprBuilder` struct stores the constraints and computes associated to a single circuit.
Every `FieldVariable` stores a shared reference to an `ExprBuilder` and it mutates the builder as needed.

When we are done building the circuit, the `ExprBuilder` has all the data necessary to build the circuit.
We can pass the `ExprBuilder` into the `FieldExpr` constructor to build an AIR.

## The Select operation

For convenience, we provide the `FieldVariable::select` function which allows for simple control flow.
For example, if `x`, `y`, and `z` are `FieldVariable`s and `s` is a flag id, then `let z = FieldVariable::select(s, x, y)` will choose between `x` and `y` based on the value of the flag.
Select is implemented as $s x + (1 - s) y$.

## The Setup instruction

Some chips have a setup instruction that verifies the modulus, along with any other relevant constants.
Mod Builder supports chips with setup instructions.
- If you call `FieldExpr::new` with `needs_setup = true` then the chip will have a setup instruction that verifies the modulus.
- If you call `FieldExpr::new_with_setup_values` with `needs_setup = true` and pass in a `Vec` of setup values, then the chip will have a setup instruction that verifies the modulus as well as the values of the constants provided in the constructor as setup values.

## Notes on flags and setup

Setup opcode is a special op that verifies the modulus, along with any other relevant constants, are correct.
There are some chips that don't need it because we hardcode the modulus. E.g. the pairing ones.
For those chips need setup, setup is derived: `setup = is_valid - sum(all_flags)`. Note that all chips have `is_valid`.
Therefore when the chip needs setup and only supports one opcode, user won't explicitly create a flag for it
and we will create a default flag for it on finalizing.

There are two independent properties of the chip built by `ExprBuilder`: whether it needs setup, and whether it supports multiple (2 for now) flags, and hence four types of chips:

| needs_setup | multi_flags | Example                                            |
| ----------- | ----------- | -------------------------------------------------- |
| true        | true        | modular, Fp2                                       |
| true        | false       | EcAdd and EcDouble                                 |
| false       | true        | Not supported, no such chips                       |
| false       | false       | Pairing ones, hardcoded modulus so no setup needed |

1. For the first type (modular and Fp2), there are two flags (e.g. `add_flag` and `sub_flag`) and `setup = is_valid - sum(all_flags)`. That is, when doing setup both flags are 0.
2. For the second type (EcAdd and EcDouble), the chip only supports one operation so technically it doesn't need a flag. But for implementation simplicity, we still create a dummy flag for it, and it's always 1 unless it's doing setup. And this `setup = is_valid - sum(all_flags)` still holds.
3. No chip is in the third type right now.
4. For the fourth type, there is no setup needed and no flags for selecting operations. Only `is_valid` is needed.

## Finalizing

The STARK backend requires the trace height to be a power of 2. Usually we pad the trace with empty (all 0) rows to make the height a power of 2. Since `is_valid` is 0 for padded rows, the constraints including interaction with memory are satisfied.
However, there are some cases that all-0 row doesn't satisfy the constraints: when the computation involves non-zero constant values:

- Some chips involve constant values, so their constraints will not be satisfied by all-0 rows.
  For these chips, we will pad the trace with a "dummy row".
  This dummy row will be created by evaluating the constraints with all-0 inputs and all-0 flags, and setting `is_valid` to 0.
  See the `FieldExpressionCoreChip::finalize` method for details.

## Note on value of `range_checker_bits`

If `range_checker_bits` is too small, the carries resulting from the product of two `FieldVariable`'s may be too large to be range checked.
If you encounter [this assert](https://github.com/openvm-org/openvm/blob/f1b484499b9c059d14949cdfaa648906757ca7aa/crates/circuits/primitives/src/bigint/utils.rs#L18C1-L24C1) failing, then this might be your problem.
Increase `range_checker_bits` and try again.

This error could also occur if you are calling `FieldVariable::int_add` or `FieldVariable::int_mul` with large constants.
These methods are not meant to be called with large constants.
Instead create a constant with `ExprBuilder::new_const` and use that.


## Usage

See the [examples section](#examples) for code examples to follow along with.

1. Start by creating an `ExprBuilderConfig`.
   Use the `check_valid` method to assert its validity.

2. Create an `ExprBuilder` with the config, let's say `let builder = ExprBuilder::new(config, range_checker_bits)`.

3. Wrap the `ExprBuilder` in an `Rc<RefCell<ExprBuilder>>` so that it can be shared between multiple `FieldVariable`s.
   (You will be passing it into each `FieldVariable` that you create.)

4. Create `FieldVariable`s using `ExprBuilder::new_input` and `ExprBuilder::new_constant`.

5. Use the `FieldVariable`s to build the expression.
   Just use the `FieldVariables` as algebraic variables to construct expressions.
   You can use the `+`, `-`, `*`, `/`, operators and the `square`, `int_add` (add a scalar), `int_mul` (multiply by a scalar) methods.
   Note that division by zero will panic.
   You can also use the `select` method to select between two `FieldVariable`s based on a flag.

6. Use the `save_output` method to save the result `FieldVariable` and to mark it as an output variable.

   Note that you can also use the `save` method to save the result `FieldVariable` without marking it as an output variable.
   Usually you don't need to do this because variables are auto-saved when there is a possibility of overflow (i.e. when the carry for any of the limbs overflows).
   But it gives greater control over how the expression is broken down into constraints, if that's needed.

7. Finally, pull out a copy of the builder as follows: `let builder = builder.borrow().clone()`, and pass it into the appropriate `FieldExpr` constructor:
    - If your chip has no setup instruction, use `FieldExpr::new(builder, range_bus, false)`.
    - If your chip has a setup instruction that only checks if the modulus is correct, use `FieldExpr::new(builder, range_bus, true)`.
    - If your chip has a setup instruction that checks the correctness of more than just the modulus, use `FieldExpr::new_with_setup_values(builder, range_bus, true, setup_values)` where `setup_values` is a `Vec<BigUint>` of values to be used in setup.
     The setup row should be filled with the modulus followed by the values in `setup_values`.

### Examples

See these examples in the elliptic curve extension code:

- [Short Weierstrass Addition Chip](https://github.com/openvm-org/openvm/blob/main/extensions/ecc/circuit/src/weierstrass_chip/add_ne/mod.rs)
- [Short Weierstrass Double Chip](https://github.com/openvm-org/openvm/blob/main/extensions/ecc/circuit/src/weierstrass_chip/double/mod.rs)
