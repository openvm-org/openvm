# How to use

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
   You can also use the `select` method to select between two `FieldVariable`s based on a flag.

   Each `FieldVariable` holds a binary syntax tree representation of its expression.

6. Use the `save_output` method to save the result `FieldVariable` and to mark it as an output variable.
   What this does is that it creates a new variable in the AIR and adds a constraint to the AIR that the new variable is equal to the expression.
   It also marks that variable as an output variable.

   Note that you can also use the `save` method to save the result `FieldVariable` without marking it as an output variable.
   Usually you don't need to do this because variables are auto-saved when there is a possibility of overflow (i.e. when the carry for any of the limbs overflows).
   But it gives greater control over how the expression is broken down into constraints, if that's needed.

7. Finally, pull out a copy of the builder as follows: `let builder = builder.borrow().clone()`, and pass it into the `FieldExpr` constructor: `FieldExpr::new(builder, range_bus, needs_setup)`.
   The `needs_setup` argument is true if the chip has a setup instruction, false otherwise.

8. If the chip has a setup instruction that checks if the modulus is correct, then you are done.
   But if your chip's setup instruction checks the correctness of more than just the modulus, you can use the `new_with_setup_values` constructor to pass in a `Vec` of `BigUint` values that will be used in setup.
   The setup row should be filled with the modulus followed by values you passed in, in the order you passed them in.

## Examples

See these examples in the elliptic curve extension code:

- [Short Weierstrass Addition Chip](https://github.com/openvm-org/openvm/blob/main/extensions/ecc/circuit/src/weierstrass_chip/add_ne.rs)
- [Short Weierstrass Double Chip](https://github.com/openvm-org/openvm/blob/main/extensions/ecc/circuit/src/weierstrass_chip/double.rs)

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
