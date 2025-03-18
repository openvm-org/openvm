# Mod Builder

The Mod Builder framework provides an easy way to build circuits that constrain arithmetic operations on modular integers.
See the [usage](#usage) section to get started and read the [specification](#specification) section for implementation details.

Note that Mod Builder assumes the proof system modulus is 31 bits.

# Usage

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

   Each `FieldVariable` holds an abstract syntax tree representation of its expression.

6. Use the `save_output` method to save the result `FieldVariable` and to mark it as an output variable.
   This creates a new variable in the AIR and adds a constraint that the new variable is equal to this expression.
   It also marks that variable as an output variable.

   Note that you can also use the `save` method to save the result `FieldVariable` without marking it as an output variable.
   Usually you don't need to do this because variables are auto-saved when there is a possibility of overflow (i.e. when the carry for any of the limbs overflows).
   But it gives greater control over how the expression is broken down into constraints, if that's needed.

7. Finally, pull out a copy of the builder as follows: `let builder = builder.borrow().clone()`, and pass it into the appropriate `FieldExpr` constructor: 
    - If your chip has no setup instruction, use `FieldExpr::new(builder, range_bus, false)`.
    - If your chip has a setup instruction that only checks if the modulus is correct, use `FieldExpr::new(builder, range_bus, true)`.
    - If your chip has a setup instruction that checks the correctness of more than just the modulus, use `FieldExpr::new_with_setup_values(builder, range_bus, true, setup_values)` where `setup_values` is a `Vec<BigUint>` of values to be used in setup.
     The setup row should be filled with the modulus followed by the values in `setup_values`.

## Examples

See these examples in the elliptic curve extension code:

- [Short Weierstrass Addition Chip](https://github.com/openvm-org/openvm/blob/main/extensions/ecc/circuit/src/weierstrass_chip/add_ne.rs)
- [Short Weierstrass Double Chip](https://github.com/openvm-org/openvm/blob/main/extensions/ecc/circuit/src/weierstrass_chip/double.rs)

# Specification

The main idea behind constraining modular arithmetic is to add a witness variable `q` and represent the operation `x <op> y = z (mod p)` (where `<op>` is one of `+`, `-`, `*`) as the constraint `x <op> y = z + q * p` in the integers.
For division, we treat `x / y = z (mod p)` as the equivalent modular congruence `x = z * y (mod p)` (we assume throughout that the modulus is prime).

There are a few more tricks.

Firstly, our proof system's modulus (the BabyBear prime) is about 31 bits, and we want to support modular arithmetic for much larger moduli (the secp256k1 modulus is 256 bits, for instance).
To do this, we represent modular integers as an array of limbs, where each limb has `limb_bits` bits.

Secondly, constraining `x <op> y = z + q * p` as integers is equivalent to setting `expr := x <op> y - z - q * p` and then constraining `expr = 0`.
We make the following optimization: while evaluating any expression involving modular integer variables, we allow the limbs of each variable to have up to `max_overflow_bits` where `max_overflow_bits > limb_bits`.
In other words, we allow the limbs to 'overflow' past the canonical `limb_bits` size.
When we evaluate `expr` in this way, we obtain an array of overflowed limbs representing `expr`.
We constrain `expr = 0` by iterating from its least significant limb to its most significant limb, carrying the overflows through to the next limb, and asserting that the final carry is 0.
To be clear, the carries are provided as witnesses, and we constrain that they are correct and that the final carry is 0.
All of this is done in the `CheckCarryModToZeroSubAir` subair, which uses the `CheckCarryToZeroSubAir` inside.
See the [bigint documentation](https://github.com/openvm-org/openvm/blob/main/crates/circuits/primitives/src/bigint/README.md) for more details.

The next subsections will explain the details of the structs used by Mod Builder.

## `ExprBuilder`

The `ExprBuilder` struct holds the actual data used to build the circuit, such as the constraints and variables, as well as a bunch of config parameters.
Each `FieldVariable` holds a shared reference to the `ExprBuilder` and they update the builder through the reference as needed, such as to add new variables or constraints.

A good place to start understanding the `ExprBuilder` is to look at the the `FieldExprCols` struct, copied below.
```rust
pub struct FieldExprCols<T> {
    pub is_valid: T,
    pub inputs: Vecs<T>,
    pub vars: Vecs<T>,
    pub q_limbs: Vecs<T>,
    pub carry_limbs: Vecs<T>,
    pub flags: Vec<T>,
}
```
These are the columns of the air that will be produced by Mod Builder.
The fields `inputs`, `vars`, `q_limbs`, and `carry_limbs` are vectors of the same length, where the `i`th element represents the `i`th constraint.
The `inputs` are the inputs of the chip, which will be read from memory, and they are the variables on which the expressions are evaluated.
The `vars` are the intermediate variables that are produced by the expressions.
We often need to use intermediate variables in the expressions so that the limbs don't get big enough to overflow the BabyBear modulus (recall that we use the optimization of allowing limbs to overflow).
Some of these variables can be marked as output variables, and these will be written to memory.
The `q_limbs` are the quotient variables (because, I suppose, `q_limbs` reads better than `qs`).
The `carry_limbs` are the carries for each limb.
The `flags` are boolean variables that can be used by the `Select` operation for primitive control flow (see the `FieldVariable` subsection for details).

The `ExprBuilder::constraints` field holds the symbolic expressions that should be evaluated to zero mod p.
The `ExprBuilder::computes` field holds the symbolic expressions that compute the variables.
For instance, if the `i`th variable is `z` which is defined by the constraint `z = x + y (mod p)` then `constraints[i] = x + y - z` and `computes[i] = x + y`.
Note that the `- q * p` part is not represented in `constraints`.

There is also `ExprBuilder::q_limbs` and `ExprBuilder::carry_limbs` which store the number of limbs in the quotient and carries for each constraint.
The reason these are not constant is that the `Select` operation may select between variables with an unequal number of limbs.

The useful methods of `ExprBuilder` are `new_input`, `new_const` and `new_flag`, which create and return input, constant and flag variables respectively (actually `new_flag` returns the `flag_id` which is the index of the flag in the `flags` vector).
The first two methods create `FieldVariable`s which are used to build expressions.

## `FieldVariable`

The `FieldVariable` struct is like a wrapper for a working variable that is used while building an expression.
It allows you to do build expressions of Type `SymbolicExpr` while using the usual `+`, `-`, `*`, `/` operators, as well as `square` and the `int_add` and `int_mul` methods for adding and multiplying by a scalar.
It also tracks the maximum possible size of the limbs of the expression and produces a new intermediate variable if necessary.

The `FieldVariable::expr` field is the actual expression that the `FieldVariable` is building.
The `FieldVariable::limb_max_abs` and `FieldVariable::max_overflow_bits` fields track the maximum possible size of the limbs of the expression.
We have that `max_overflow_bits = ceil(log_2(limb_max_abs))`.
These upper bounds are maintained by using worst-case estimates for the values of each limb of a variable.
That is, `FieldVariable` assumes that all limbs of an input variable have the value `2^limb_bits - 1`.
The limb sizes of intermediate variables are computed recursively based on the operations performed on the input variables.

Whenever performing an operation on a `FieldVariable` would cause the worst-case limb size to be too large, a new intermediate variable is created in the `ExprBuilder`, and it is constrained to the current expression by appending to the `ExprBuilder::constraints` vector.
We call this a 'save'.
A save effictively resets the limb size bounds to `limb_bits`, since the intermediate variable witnesses are range checked to be in `[0, 2^limb_bits)`.
See the [`FieldVariable::save_if_overflow`](https://github.com/openvm-org/openvm/blob/f1b484499b9c059d14949cdfaa648906757ca7aa/crates/circuits/mod-builder/src/field_variable.rs#L86C1-L110C6) method for details.

A save can also be triggered manually by calling `FieldVariable::save`, though this is not typically needed.

A division is a bit trickier.
Our proof system doesn't support division, so our constraints cannot have division.
When a division is performed, let's say `x / y`, then we save the expression which creates a new variable `z` and we add the constraint `z * y = x` as well as the 'compute' expression `z = x / y`.
Note that we allow division in compute expressions, since these are used in trace generation and evaluated as bigints.
We can then continue using `z` in expressions as any other variable.

Besides the arithmetic operations, `FieldVariable` also supports the `Select` operation which allows for conditionally selecting between two `FieldVariable`s based on a boolean flag.
Use `ExprBuilder::new_flag` to create a flag and pass it into `FieldVariable::select` to perform a select.

## `SymbolicExpr`

The `SymbolicExpr` enum is an expression tree that is built when using the `FieldVariable` methods.
It is used to build both the constraints and computes of the `ExprBuilder`.
In `SymbolicExpr`s that represent constraints, division is not allowed.

The `SymbolicExpr` enum has a bunch of methods that recursively calculate certain attributes of the expression tree.
The simplest ones are the `SymbolicExpr::evaluate_*` methods which evaluate the expression tree as various types.
We have `SymbolicExpr::evaluate_bigint`, `SymbolicExpr::evaluate_overflow_isize`, and `SymbolicExpr::evaluate_overflow_expr` that return a `BigInt`, `OverflowInt<isize>`, and `OverflowInt<AB::Expr>` respectively.
These should only be called on `SymbolicExpr`s that represent constraints.

Similarly, `SymbolicExpr::compute` recursively evaluates the expression tree as a `BigUint` and keeps the value in `[0, p)` by taking modulo `p`.
This should only be called on `SymbolicExpr`s that represent computes.

The `SymbolicExpr::max_abs` method returns the maximum possible positive value and maximum possible absolute negative value of the expression.
That is, if `(r, l) = expr.max_abs(p)` then `l,r >= 0` and `-l <= expr <= r`.
This is a helper method used in `SymbolicExpr::constraint_limbs`.

The `SymbolicExpr::expr_limbs` method is also a helper method only used in `SymbolicExpr::constraint_limbs`.

The `SymbolicExpr::constraint_limbs` method returns `(q_limbs, carry_limbs)` where `q_limbs` is the number of limbs in the quotient and `carry_limbs` is the number of limbs in the carry of the constraint `self.expr - q * p = 0`.
It is used in `ExprBuilder::set_constraint` to update `ExprBuilder::q_limbs` and `ExprBuilder::carry_limbs` when adding a new constraint.
It is also used in `FieldVariable::save_if_overflow` to calculate the maximum possible limb size of the intermediate variable.

The `SymbolicExpr::constraint_limb_max_abs` method returns the maximum possible size, in bits, of each limb in `self.expr - q * p`.

The `SymbolicExpr::constraint_carry_bits_with_pq` method returns the maximum possible size, in bits, of each carry in `self.expr - q * p`.
It is used in `FieldVariable::div` to decide if we need to save the dividend and divisor.
This method calls `SymbolicExpr::constraint_limb_max_abs` which recurses through the expression tree to calculate the maximum possible limb size. 
Note that in `FieldVariable::save_if_overflow` we avoid calling `SymbolicExpr::constraint_carry_bits_with_pq` since the maximum possible limb size is already tracked in `FieldVariable`.

## `FieldExpr`

The `FieldExpr` struct implements the air traits and is used as a subair in `FieldExpressionCoreAir`.

There are two constructors for `FieldExpr`: `new` and `new_with_setup_values`, depending on whether the chip's setup row checks the correctness of more values than just the modulus.
This is used in the elliptic curve extension to verify that the curve constants are correct.

The `FieldExpr::eval` method calls `SymbolicExpr::evaluate_overflow_expr` to evaluate the expression as an `OverflowInt<AB::Expr>` and then passes it to the `CheckCarryModToZeroSubAir` sub air.
It also adds constraints asserting that the setup row consists of the modulus followed by the values in `setup_values`.

The variables are range checked to `limb_bits` in `FieldExpressionCoreAir::eval` while the quotients are range checked to `[-2^limb_bits, 2^limb_bits)` in `CheckCarryModToZeroSubAir::eval` and the carries are range checked in `CheckCarryToZeroSubAir::eval`.
See the [BigInt documentation](https://github.com/openvm-org/openvm/blob/main/crates/circuits/primitives/src/bigint/README.md) for more details.


## `FieldExpressionCoreAir` / `FieldExpressionCoreChip`

These are the vm air and vm chip structs.

If the chip needs setup and only supports one opcode, the chip will create a default flag for it.
See the [Notes on flags and setup](#notes-on-flags-and-setup) subsection for more details.

See also the [Finalizing](#finalizing) subsection for more details on how the trace is padded.


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
