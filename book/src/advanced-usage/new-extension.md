# Creating a New Extension

Extensions in OpenVM let you introduce additional functionality without disrupting the existing system. Consider, for example, an extension that provides two new operations on `u32` values: one that squares a number, and another that multiplies it by three. With such an extension:

1. You define functions like `square(x: u32) -> u32` and `mul3(x: u32) -> u32` for use in guest code.
2. When the compiler encounters these functions, it generates corresponding custom [RISC-V instructions](https://github.com/openvm-org/openvm/blob/main/docs/specs/RISCV.md).
3. During the transpilation phase, these custom instructions are translated into OpenVM instructions.
4. At runtime, the OpenVM program sends these new instructions to a specialized [chip](https://github.com/openvm-org/openvm/blob/main/docs/specs/circuit.md) that computes the results and ensures their correctness.

This modular architecture means the extension cleanly adds new capabilities while leaving the rest of OpenVM untouched. The entire system, including the extension’s operations, can still be proven correct.

Conceptually, a new extension consists of three parts:

- **Guest**: High-level Rust code that defines and uses the new operations.
- **Transpiler**: Logic that converts custom RISC-V instructions into corresponding OpenVM instructions.
- **Circuit**: The special chips that enforce correctness of instruction execution through polynomial constraints.

## Guest

In the guest component, your goal is to produce the instructions that represent the new operations. When you want to perform an operation (for example, “calculate _this_ new function of _these_ arguments and write the result _here_”), you generate a custom instruction. You can use the helper macros `custom_insn_r!` and `custom_insn_i!` from `openvm_platform` to ensure these instructions follow the [RISC-V specification](https://riscv.org/specifications/ratified/). For more details, see the RISC-V [contributor documentation](https://github.com/openvm-org/openvm/blob/main/docs/specs/RISCV.md).

## Transpiler

The transpiler maps the newly introduced RISC-V instructions to their OpenVM counterparts. To achieve this, implement a struct that provides the `TranspilerExtension` trait, which includes:

```rust
fn process_custom(&self, instruction_stream: &[u32]) -> Option<(Instruction<F>, usize)>;
```

This function checks if the given instruction stream starts with one of your custom instructions. If so, it returns the corresponding OpenVM instruction and how many 32-bit words were consumed. If not, it returns `None`.

Note that almost always the valid instruction consists of a single 32-bit RISC-V word (so whenever `Some(_, sz)` is returned, `sz` is 1), but in general this may not be the case.

## Circuit

The circuit component is where the extension’s logic is enforced in a zero-knowledge proof context. Here, you create a chip that:

- Implements the computing logic, so that the output always corresponds to the correct result of the new operation. The chip has access to the memory shared with the other chips from the VM via [our special architecture](https://github.com/openvm-org/openvm/blob/main/docs/specs/ISA.md).
- Properly constrains all the inputs, outputs and intermediate variables using polynomial equations in such a way that there is no way to fill these variables with values that correspond to an incorrect output while fitting the constraints.

For more technical details on writing circuits and constraints, consult the OpenVM [contributor documentation](https://github.com/openvm-org/openvm/blob/main/docs/specs/README.md), which provides specifications and guidelines for integrating your extension into the OpenVM framework.
