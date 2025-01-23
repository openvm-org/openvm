## Notes on flags and setup

Setup opcode is a special op that verifies the modulus is correct.
There are some chips that don't need it because we hardcode the modulus. E.g. the pairing ones.
For those chips need setup, setup is derived: `setup = is_valid - sum(all_flags)`. Note that all chips have `is_valid`.
Therefore when the chip needs setup and only supports one opcode, user won't explicitly create a flag for it
and we will create a default flag for it on finalizing.

There are two independent properties of the chip built by `ExprBuilder`: whether it needs setup, and whether it supports multiple (2 for now) flags, and hence four types of chips:

| needs_setup | multi_flags | Example |
|-------------|-------------|---------|
| true        | true        | modular, Fp2 |
| true        | false       | EcAdd and EcDouble |
| false       | true        | Not supported, no such chips |
| false       | false       | Pairing ones, hardcoded modulus so no setup needed |


1. For the first type (modular and Fp2), there are two flags (e.g. `add_flag` and `sub_flag`) and `setup = is_valid - sum(all_flags)`. That is, when doing setup both flags are 0.
2. For the second type (EcAdd and EcDouble), the chip only supports one operation so technically it doesn't need a flag. But for implementation simplicity, we still create a dummy flag for it, and it's always 1 unless it's doing setup. And this `setup = is_valid - sum(all_flags)` still holds.
3. No chip is in the third type right now.
4. For the fourth type, there is no setup needed and no flags for selecting operations. Only `is_valid` is needed.

### Finalizing

The STARK backend requires the trace height to be a power of 2. Usually we pad the trace with empty (all 0) rows to make the height a power of 2. Since `is_valid` is 0 for padded rows, the constraints including interaction with memory are satisfied.
However, there are some cases that all-0 row doesn't satisfy the constraints: when the computation involves non-zero constant values:

- Some pairing chips involves adding a small constant value (like `1`). But since pairing chips don't have any flags, we will pad the trace with the last valid row and set `is_valid` to 0.
- EcDouble for short Weierstrass curve: `y^2 = x^3 + ax + b` where `a != 0`. Last valid row with `is_valid = 0` won't work as in that case `setup = is_valid - sum(all_flags) = 0 - 1 = -1` is not a bool (0/1). So we will pad the trace with the first valid row (which is a setup row) and set `is_valid` to 0.

## Example usage

Ec Double chip:

```rust
let x1 = ExprBuilder::new_input(builder.clone());
let y1 = ExprBuilder::new_input(builder.clone());
let nom = (x1.clone() * x1.clone()).scalar_mul(3);
let denom = y1.scalar_mul(2);
let lambda = nom / denom;
let mut x3 = lambda.clone() * lambda.clone() - x1.clone() - x1.clone();
x3.save();
let mut y3 = lambda * (x1 - x3) - y1;
y3.save();
```
