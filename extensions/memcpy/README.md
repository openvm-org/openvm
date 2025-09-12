# OpenVM Memcpy Extension

This extension provides a custom RISC-V instruction `memcpy_loop` that optimizes memory copy operations by handling different alignment shifts efficiently.

## Custom Instruction: `memcpy_loop shift`

### Format
```
memcpy_loop shift
```

Where `shift` is an immediate value (0, 1, 2, or 3) representing the byte alignment shift.

### RISC-V Encoding
- **Opcode**: `0x73` (custom opcode)
- **Funct3**: `0x0` (custom funct3)
- **Immediate**: 12-bit signed immediate for shift value
- **Format**: I-type instruction

### Usage
The `memcpy_loop` instruction is designed to replace repetitive shift-handling code in memcpy implementations. Instead of having separate code blocks for each shift value, you can use a single instruction:

```assembly
# Instead of this repetitive code:
.Lshift_1:
    lw      a5, 0(a4)
    sb      a5, 0(a3)
    srli    a1, a5, 8
    sb      a1, 1(a3)
    # ... more shift handling code

# You can use:
memcpy_loop 1    # Handles shift=1 case
```

Note that you must define `memcpy_loop` before using it. For example, in [memcpy.s](../../crates/toolchain/openvm/src/memcpy.s) it is defined at the beginning of the assembly code as follows:
```assembly
.macro memcpy_loop shift
		.word 0x00000072 | (\shift << 12)  # opcode 0x72 + shift in immediate field (bits 12-31)
```

### Benefits
1. **Code Size Reduction**: Eliminates repetitive shift-handling code
2. **Performance**: Optimized implementation in the circuit layer
3. **Maintainability**: Single instruction handles all shift cases
4. **Verification**: Zero-knowledge proof ensures correct execution

## Implementation Details

### Circuit Layer
The instruction is implemented in the `MemcpyIterationAir` circuit which:
- Reads 4 words (16 bytes) from memory
- Applies the specified shift to combine words
- Writes the result to the destination
- Handles all shift values (0, 1, 2, 3) efficiently

### Transpiler Extension
The `MemcpyTranspilerExtension` translates the RISC-V instruction into OpenVM's internal format:
- Parses I-type instruction format
- Validates shift value (0-3)
- Converts to OpenVM instruction with shift as operand

### Example Usage
See `example_memcpy_optimized.s` for a complete example showing how to use the custom instruction to optimize a memcpy implementation.

## Building and Testing

### Compilation
```bash
# Build the extension
cargo build --package openvm-memcpy-circuit --package openvm-memcpy-transpiler

# Check for compilation errors
cargo check --package openvm-memcpy-circuit --package openvm-memcpy-transpiler
```

### Integration
To use this extension in your OpenVM project:

1. Add the transpiler extension to your OpenVM configuration
2. Use the `memcpy_loop` instruction in your RISC-V assembly
3. The circuit will handle the execution and verification

## Architecture

```
RISC-V Assembly → Transpiler Extension → OpenVM Instruction → MemcpyIterationAir → Execution
```

The extension provides:
- **Transpiler**: `extensions/memcpy/transpiler/` - Translates RISC-V to OpenVM
- **Circuit**: `extensions/memcpy/circuit/` - Implements the instruction logic


# References

- [OpenVM Documentation](../../docs/README.md)
- [RISC-V Instruction Set Manual](https://riscv.org/technical/specifications/)
