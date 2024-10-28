# axVM Instruction Set Architecture

# Table of Contents

- [axVM Architecture](#architecture)
- [axVM Instruction Set](#axvm-instruction-set)
- [RISC-V Custom Instructions](#risc-v-custom-instructions)
- [RISC-V to axVM Transpilation](#risc-v-to-axvm-transpilation)

# axVM Architecture

This design is adapted from [Valida](https://github.com/valida-xyz/valida-compiler/issues/2) with changes to the
instruction format suggested by Max Gillet to enable easier compatibility with other existing ISAs.

## Instruction format

Instructions are encoded as a global opcode (field element) followed by `NUM_OPERANDS = 7` operands (field elements): `opcode, a, b, c, d, e, f, g`. An instruction does not need to use all operands, and trailing unused operands should be set to zero.

## Program ROM

Our VM operates under the Harvard architecture, where program code is stored separately from main
memory. Code is addressed by any field element in range `[0, 2^PC_BITS)` where `PC_BITS = 30`.

There is a single special purpose register `pc` for the program counter of type `F` which stores the location of the instruction being executed.
(We may extend `pc` to multiple field elements to increase the program address space size in the future.)

The program code is committed as a cached trace. The validity of the program code and its cached trace must be checked outside of ZK. A valid program code must have all instructions stored at locations in range `[0, 2^PC_BITS)`. While the instructions can be stored at any locations, we will by default follow RISC-V in storing instructions at multiples of `DEFAULT_PC_STEP = 4`.

## Memory

Memory is comprised of addressable cells, each cell containing a single field element.
Instructions of the VM may access (read or write) memory
as single cells or as a contiguous list of cells. Such a contiguous list is called a _block_, and
an memory access (read/write) to a block is a _block access_.
The architecture distinguishes between block accesses of different sizes as this has significant performance implications.
The number of cells in a block access is restricted to powers of two, of which the following are supported: 1, 2, 4, 8, 16, 32, 64. Block accesses must be
aligned, meaning that in a block access of size $N$, the starting pointer must be divisible by $N$ (as an integer).

We also leave open the possibility in the future that different address spaces (see below) can be dedicated to handling
data with certain block sizes, effectively declaring a word-size for that address space, but this is not currently
implemented. At present there are two types of blocks we have in mind

- **[FVEC]** A block consisting of `[F; N]` arbitrary field elements.
- **[LIMB]** A block consisting of `[F; N]` field elements, where each field element has as its canonical representation
  a limb in `[0, 2^LIMB_BITS)`. This would emulate a word in RISC-V memory.

While not relevant to the ISA itself, the ZK circuit implementation does usually represent a block `[F; N]` as `N`
contiguous field elements in the same row of the trace matrix.

## Immediate Values

Immediate values are treated as single field elements. Our VM cannot represent operand values that are greater than the
prime $p$ and cannot distinguish between $0$ and $p$ (or any two integers whose difference is a multiple of $p$).
Therefore, any immediate values greater than or equal to $p$ need to be expanded into smaller values.

## Registers

Our zkVM treats general purpose registers simply as pointers to a separate address space, which is also comprised of
addressable cells. Registers are represented using the [LIMB] format with `LIMB_BITS = 8`.

## Notation

The following notation is used throughout this document:

### Operand values

`a, b, c, d, e, f, g` denote the value encoded in the corresponding operand of the current instruction.

### Program counter

`pc` denotes the value of the current program counter.

### Addressing

We support different address spaces of memory.

- We use `[a]_d` to denote the single-cell value at pointer location `a` in address space `d`. This is a single
  field element.
- We use `[a:N]_d` to denote the slice `[a..a + N]_d` -- this is a length-`N` array of field elements.

We will always have the following fixed address spaces:

| Address Space | Name          |
| ------------- | ------------- |
| `0`           | Immediates    |
| `1`           | Registers     |
| `2`           | User Memory   |
| `3`           | User IO       |
| `4`           | Native Kernel |

Address space `0` is not a real address space: it is reserved for denoting immediates: We define `[a]_0 = a`.

The number of address spaces supported is a configurable constant of the VM. The address spaces in `[as_offset, as_offset + 2^addr_space_max_bits)` are supported. By default `as_offset = 1` to preclude address space `0`.

The size (= number of pointers) of each address space is also a configurable constant of the VM.
The pointers can have values in `[0, 2^pointer_max_bits)`. We require `addr_space_max_bits, pointer_max_bits <= F::bits() - 2` due to a sorting argument.

> A memory cell in any address space is always a field element, but the VM _may_ later impose additional bit size
> constraints on certain address spaces (e.g., everything in address space `2` must be a byte).

# axVM Instruction Set

All instruction types are divided into classes, mostly based on purpose and nature of the operation (e.g., ALU instructions, U256 instructions, Modular arithmetic instructions, etc).
Instructions within each class are usually handled by the same chip, but this is not always the case (for example, if
one of the operations requires much more trace columns than all others).
Internally, certain non-intersecting ranges of opcodes (which are internally just a `usize`) are distributed among the
enabled operation classes, so that there is no collision between the classes.

Operands marked with `_` means they are not used and should be set to zero. Trailing unused operands should also be set to zero.
Unless otherwise specified, instructions will by default set `to_pc = from_pc + DEFAULT_PC_STEP`.

The architecture is independent of RISC-V, but for transpilation purposes we specify additional information such as the RISC-V Opcode (7-bit), `funct3` (3-bit), and `funct7` (7-bit) or `imm` fields depending on the RISC-V instruction type.

We will use the following notation:

- `u32(x)` where `x: [F; 4]` consists of 4 bytes will mean the casting from little-endian bytes in 2's complement to unsigned 32-bit integer.
- `i32(x)` where `x: [F; 4]` consists of 4 bytes will mean the casting from little-endian bytes in 2's complement to signed 32-bit integer.
- `sign_extend` means sign extension of bits in 2's complement.
- `i32(c)` where `c` is a field element will mean `c.as_canonical_u32()` if `c.as_canonical_u32() < F::modulus() / 2` or `-c.as_canonical_u32()` otherwise.
- `decompose(c)` where `c` is a field element means `c.as_canonical_u32().to_le_bytes()`.

## System

| Name      | Operands  | Description                                                                                                        |
| --------- | --------- | ------------------------------------------------------------------------------------------------------------------ |
| TERMINATE | `_, _, c` | Terminates execution with exit code `c`. Sets `to_pc = from_pc`.                                                   |
| PHANTOM   | `_, _, c` | Sets `to_pc = from_pc + DEFAULT_PC_STEP`. The operand `c` determines which phantom instruction (see below) is run. |

## RV32IM Support

While the architecture allows creation of VMs without RISC-V support, we define a set of instructions that are meant to be transpiled from RISC-V instructions such that the resulting VM is able to run RISC-V ELF binaries. We use \_RV32 to specify that the operand parsing is specifically targeting 32-bit RISC-V registers.

All instructions below assume that all memory cells in address spaces `1` and `2` are field elements that are in range `[0, 2^LIMB_BITS)` where `LIMB_BITS = 8`. The instructions must all ensure that any writes will uphold this constraint.

`x0` handling: Unlike in RISC-V, the instructions will **not** discard writes to `[0:4]_1` (corresponding to register `x0`). A valid transpilation of a RISC-V program can be inspected to have the properties:

1. `[0:4]_1` starts with all zeros.
2. No instruction in the program writes to `[0:4]_1`.

### ALU

In all ALU instructions, the operand `d` is fixed to be `1`. The operand `e` must be either `0` or `1`. When `e = 0`, the `c` operand is expected to be of the form `F::from_canonical_u32(c_i16 as i24 as u24 as u32)` where `c_i16` is type `i16`. In other words we take signed 16-bits in two's complement, sign extend to 24-bits, consider the 24-bits as unsigned integer, and convert to field element. In the instructions below, `[c:4]_0` should be interpreted as `c_i16 as i32` sign extended to 32-bits.

| Name      | Operands    | Description                                                                                              |
| --------- | ----------- | -------------------------------------------------------------------------------------------------------- |
| ADD_RV32  | `a,b,c,1,e` | `[a:4]_1 = [b:4]_1 + [c:4]_e`. Overflow is ignored and the lower 32-bits are written to the destination. |
| SUB_RV32  | `a,b,c,1,e` | `[a:4]_1 = [b:4]_1 - [c:4]_e`. Overflow is ignored and the lower 32-bits are written to the destination. |
| XOR_RV32  | `a,b,c,1,e` | `[a:4]_1 = [b:4]_1 ^ [c:4]_e`                                                                            |
| OR_RV32   | `a,b,c,1,e` | `[a:4]_1 = [b:4]_1 \| [c:4]_e`                                                                           |
| AND_RV32  | `a,b,c,1,e` | `[a:4]_1 = [b:4]_1 & [c:4]_e`                                                                            |
| SLL_RV32  | `a,b,c,1,e` | `[a:4]_1 = [b:4]_1 << [c:4]_e`                                                                           |
| SRL_RV32  | `a,b,c,1,e` | `[a:4]_1 = [b:4]_1 >> [c:4]_e`                                                                           |
| SRA_RV32  | `a,b,c,1,e` | `[a:4]_1 = [b:4]_1 >> [c:4]_e` MSB extends                                                               |
| SLT_RV32  | `a,b,c,1,e` | `[a:4]_1 = i32([b:4]_1) < i32([c:4]_e) ? 1 : 0`                                                          |
| SLTU_RV32 | `a,b,c,1,e` | `[a:4]_1 = u32([b:4]_1) < u32([c:4]_e) ? 1 : 0`                                                          |

### Load/Store

For all load/store instructions, we assume the operand `c` is in `[0, 2^16)`, and we fix address spaces `d = 1`.
The address space `e` can be any [valid address space](#addressing).
We will use shorthand `r32{c}(b) := i32([b:4]_1) + sign_extend(decompose(c)[0:2])` as `i32`. This means we interpret `c` as the 2's complement encoding of a 16-bit integer, sign extend it to 32-bits, and then perform signed 32-bit addition with the value of the register `[b:4]_1`.
Memory access to `ptr: i32` is only valid if `0 <= ptr < 2^addr_max_bits`, in which case it is an access to `F::from_canonical_u32(ptr as u32)`.

All load/store instructions always do block accesses of block size `4`, even for LOADB_RV32, STOREB_RV32.

| Name        | Operands    | Description                                                                                                                    |
| ----------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------ |
| LOADB_RV32  | `a,b,c,1,e` | `[a:4]_1 = sign_extend([r32{c}(b):1]_e)` Must sign-extend the byte read from memory, which is represented in 2’s complement.   |
| LOADH_RV32  | `a,b,c,1,e` | `[a:4]_1 = sign_extend([r32{c}(b):2]_e)` Must sign-extend the number read from memory, which is represented in 2’s complement. |
| LOADW_RV32  | `a,b,c,1,e` | `[a:4]_1 = [r32{c}(b):4]_e`                                                                                                    |
| LOADBU_RV32 | `a,b,c,1,e` | `[a:4]_1 = zero_extend([r32{c}(b):1]_e)` Must zero-extend the number read from memory.                                         |
| LOADHU_RV32 | `a,b,c,1,e` | `[a:4]_1 = zero_extend([r32{c}(b):2]_e)` Must zero-extend the number read from memory.                                         |
| STOREB_RV32 | `a,b,c,1,e` | `[r32{c}(b):1]_e <- [b:1]_1`                                                                                                   |
| STOREH_RV32 | `a,b,c,1,e` | `[r32{c}(b):2]_e <- [b:2]_1`                                                                                                   |
| STOREW_RV32 | `a,b,c,1,e` | `[r32{c}(b):4]_e <- [b:4]_1`                                                                                                   |

### Branch/Jump/Upper Immediate

For branch instructions, we fix `d = e = 1`. For jump instructions, we fix `d = 1`.

| Name       | Operands      | Description                                                                                                                                                                                                                                                                                                                       |
| ---------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| BEQ_RV32   | `a,b,c,1,1`   | `if([a:4]_1 == [b:4]_1) pc += c`                                                                                                                                                                                                                                                                                                  |
| BNE_RV32   | `a,b,c,1,1`   | `if([a:4]_1 != [b:4]_1) pc += c`                                                                                                                                                                                                                                                                                                  |
| BLT_RV32   | `a,b,c,1,1`   | `if(i32([a:4]_1) < i32([b:4]_1)) pc += c`                                                                                                                                                                                                                                                                                         |
| BGE_RV32   | `a,b,c,1,1`   | `if(i32([a:4]_1) >= i32([b:4]_1)) pc += c`                                                                                                                                                                                                                                                                                        |
| BLTU_RV32  | `a,b,c,1,1`   | `if(u32([a:4]_1) < u32([b:4]_1)) pc += c`                                                                                                                                                                                                                                                                                         |
| BGEU_RV32  | `a,b,c,1,1`   | `if(u32([a:4]_1) >= u32([b:4]_1)) pc += c`                                                                                                                                                                                                                                                                                        |
| JAL_RV32   | `a,_,c,1,_,f` | `if(f!=0) [a:4]_1 = decompose(pc+4); pc += c`. The operand `f`is assumed to be boolean. The`pc`increment is always done regardless of`f`'s value. Here `i32(c)`must be in`[-2^24, 24)`.                                                                                                                                           |
| JALR_RV32  | `a,b,c,1,_,f` | `if(f!=0) [a:4]_1 = decompose(pc+4); pc = F::from_canonical_u32(i32([b:4]_1) + sign_extend(decompose(c)[0:2]) as u32)`. Constrains that `i32([b:4]_1) + i32(c) is in [0, 2^PC_BITS)`. Here `i32(c)` must be in `[0, 2^16)`. The operand `f`is assumed to be boolean. The `pc` assignment is always done regardless of`f`'s value. |
| LUI_RV32   | `a,_,c,1,_,1` | `[a:4]_1 = u32(c) << 12`. Here `i32(c)` must be in `[0, 2^20)`.                                                                                                                                                                                                                                                                   |
| AUIPC_RV32 | `a,_,c,1,_,_` | `[a:4]_1 = decompose(pc) + (decompose(c) << 12)`. Here `i32(c)` must be in `[0, 2^20)`.                                                                                                                                                                                                                                           |

For branch and JAL_RV32 instructions, the instructions assume that the operand `i32(c)` is in `[-2^24,2^24)`. The assignment `pc += c` is done as field elements.
In valid programs, the `from_pc` is always in `[0, 2^PC_BITS)`. We will only use base field `F` satisfying `2^PC_BITS + 2*2^24 < F::modulus()` so `to_pc = from_pc + c` is only valid if `i32(from_pc) + i32(c)` is in `[0, 2^PC_BITS)`.

For JALR_RV32, we treat `c` in `[0, 2^16)` as a raw encoding of 16-bits. Within the instruction, the 16-bits are interpreted in 2's complement and sign extended to 32-bits. Then it is added to the register value `i32([b:4]_1)`, where 32-bit overflow is ignored. The instruction is only valid if the resulting `i32` is in range `[0, 2^PC_BITS)`. The result is then cast to `u32` and then to `F` and assigned to `pc`.

For LUI_RV32 and AUIPC_RV32, we are treating `c` in `[0, 2^20)` as a raw encoding of 20-bits. The instruction does not need to interpret whether the register is signed or unsigned.
For AUIPC_RV32, the addition is treated as unchecked `u32` addition since that is the same as `i32` addition at the bit level.

Note that AUIPC_RV32 does not have any condition for the register write.

### Multiplication Extension

For multiplication extension instructions, we fix `d = 1`.
MUL_RV32 performs an 32-bit×32-bit multiplication and places the lower 32 bits in the
destination cells. MULH_RV32, MULHU_RV32, and MULHSU_RV32 perform the same multiplication but return
the upper 32 bits of the full 2×32-bit product, for signed×signed, unsigned×unsigned, and
signed×unsigned multiplication respectively.

DIV_RV32 and DIVU_RV32 perform signed and unsigned integer division of 32-bits by 32-bits. REM_RV32
and REMU_RV32 provide the remainder of the corresponding division operation. Integer division is defined by `dividend = q * divisor + r` where `0 <= |r| < |divisor|` and either `sign(r) = sign(divisor)` or `r = 0`.

Below `x[n:m]` denotes the bits from `n` to `m` inclusive of `x`.

| Name        | Operands  | Description                                                                                                                                                                                                |
| ----------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MUL_RV32    | `a,b,c,1` | `[a:4]_1 = ([b:4]_1 * [c:4]_1)[0:3]`                                                                                                                                                                       |
| MULH_RV32   | `a,b,c,1` | `[a:4]_1 = (sign_extend([b:4]_1) * sign_extend([c:4]_1))[4:7]`. We sign extend `b` and `c` into 8-limb (i.e. 64-bit) integers                                                                              |
| MULHSU_RV32 | `a,b,c,1` | `[a:4]_1 = (sign_extend([b:4]_1) * zero_extend([c:4]_1))[4:7]`. We sign extend                                                                                                                             |
| MULHU_RV32  | `a,b,c,1` | `[a:4]_1 = (zero_extend([b:4]_1) * zero_extend([c:4]_1))[4:7]`                                                                                                                                             |
| DIV_RV32    | `a,b,c,1` | `[a:4]_1 = [b:4]_1 / [c:4]_1` integer division. Division by zero: if `i32([c:4]_1) = 0`, set `i32([a:4]_1) = -1`. Overflow: if `i32([b:4]_1) = -2^31` and `i32([c:4]_1) = -1`, set `i32([a:4]_1) = -2^31`. |
| DIVU_RV32   | `a,b,c,1` | `[a:4]_1 = [b:4]_1 / [c:4]_1` integer division. Division by zero: if `u32([c:4]_1) = 0`, set `u32([a:4]_1) = 2^32 - 1`.                                                                                    |
| REM_RV32    | `a,b,c,1` | `[a:4]_1 = [b:4]_1 % [c:4]_1` integer remainder. Division by zero: if `i32([c:4]_1) = 0`, set `[a:4]_1 = [b:4]_1`. Overflow: if `i32([b:4]_1) = -2^31` and `i32([c:4]_1) = -1`, set `[a:4]_1 = 0`.         |
| REMU_RV32   | `a,b,c,1` | `[a:4]_1 = [b:4]_1 % [c:4]_1` integer remainder. Division by zero: if `u32([c:4]_1) = 0`, set `[a:4]_1 = [b:4]_1`.                                                                                         |

### System Calls

There are currently no system calls. System calls are used when the ISA and system are customized separately.
Since axVM controls both the ISA and the underlying virtual machine, we use custom opcodes directly whenever possible.

<!--
Currently we have no need for `ECALLBREAK`, but we include it for future use.

| Name       | Operands | Description                                                                                                                                                                                              |
| ---------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ECALLBREAK | `_,_,c`  | This instruction has no operands except immediate `c = 0x0` or `0x1`. `c = 0x0` is ECALL. Custom functionality determined by reading register values. `c = 0x1` is EBREAK. Transfer control to debugger. |
-->

## RV32 Intrinsics

RV32 intrinsics are custom axVM opcodes that are designed to be compatible with the RV32 architecture.
We continue to use \_RV32 to specify that the operand parsing is specifically targeting 32-bit RISC-V registers.

All instructions below assume that all memory cells in address spaces `1` and `2` are field elements that are in range `[0, 2^LIMB_BITS)` where `LIMB_BITS = 8`. The instructions must all ensure that any writes will uphold this constraint.

We use the same notation for `r32{c}(b) := i32([b:4]_1) + sign_extend(decompose(c)[0:2])` as in [`LOADW_RV32` and `STOREW_RV32`](#loadstore).

### User IO

| Name            | Operands    | Description                                                                                              |
| --------------- | ----------- | -------------------------------------------------------------------------------------------------------- |
| HINTSTOREW_RV32 | `_,b,c,1,2` | `[r32{c}(b):4]_2 = next 4 bytes from hint stream`. Only valid if next 4 values in hint stream are bytes. |
| REVEAL_RV32     | `a,b,c,1,3` | Pseudo-instruction for `STOREW_RV32 a,b,c,1,3` writing to the user IO address space `3`.                 |

### Hashes

| Name           | Operands    | Description                                                                                                       |
| -------------- | ----------- | ----------------------------------------------------------------------------------------------------------------- |
| KECCAK256_RV32 | `a,b,c,1,e` | `[r32{0}(a):32]_e = keccak256([r32{0}(b)..r32{0}(b)+r32{0}(c)]_e)`. Performs memory accesses with block size `4`. |

### 256-bit Integers

The 256-bit ALU intrinsic instructions perform operations on 256-bit signed/unsigned integers where integer values are read/written from/to memory in address space `2`. The address space `2` pointer locations are obtained by reading register values in address space `1`. Note that these instructions are not the same as instructions on 256-bit registers.

For each instruction, the operand `d` is fixed to be `1` and `e` is fixed to be `2`.
Each instruction performs block accesses with block size `4` in address space `1` and block size `32` in address space `2`.

#### 256-bit ALU

| Name         | Operands    | Description                                                                                                                          |
| ------------ | ----------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| ADD256_RV32  | `a,b,c,1,2` | `[r32{0}(a):32]_2 = [r32{0}(b):32]_2 + [r32{0}(c):32]_2`. Overflow is ignored and the lower 256-bits are written to the destination. |
| SUB256_RV32  | `a,b,c,1,2` | `[r32{0}(a):32]_2 = [r32{0}(b):32]_2 - [r32{0}(c):32]_2`. Overflow is ignored and the lower 256-bits are written to the destination. |
| XOR256_RV32  | `a,b,c,1,2` | `[r32{0}(a):32]_2 = [r32{0}(b):32]_2 ^ [r32{0}(c):32]_2`                                                                             |
| OR256_RV32   | `a,b,c,1,2` | `[r32{0}(a):32]_2 = [r32{0}(b):32]_2 \| [r32{0}(c):32]_2`                                                                            |
| AND256_RV32  | `a,b,c,1,2` | `[r32{0}(a):32]_2 = [r32{0}(b):32]_2 & [r32{0}(c):32]_2`                                                                             |
| SLL256_RV32  | `a,b,c,1,2` | `[r32{0}(a):32]_2 = [r32{0}(b):32]_2 << [r32{0}(c):32]_2`                                                                            |
| SRL256_RV32  | `a,b,c,1,2` | `[r32{0}(a):32]_2 = [r32{0}(b):32]_2 >> [r32{0}(c):32]_2`                                                                            |
| SRA256_RV32  | `a,b,c,1,2` | `[r32{0}(a):32]_2 = [r32{0}(b):32]_2 >> [r32{0}(c):32]_2` MSB extends                                                                |
| SLT256_RV32  | `a,b,c,1,2` | `[r32{0}(a):32]_2 = i256([r32{0}(b):32]_2) < i256([r32{0}(c):32]_2) ? 1 : 0`                                                         |
| SLTU256_RV32 | `a,b,c,1,2` | `[r32{0}(a):32]_2 = u256([r32{0}(b):32]_2) < u256([r32{0}(c):32]_2) ? 1 : 0`                                                         |

#### 256-bit Branch

| Name         | Operands    | Description                                                    |
| ------------ | ----------- | -------------------------------------------------------------- |
| BEQ256_RV32  | `a,b,c,1,2` | `if([r32{0}(a):32]_2 == [r32{0}(b):32]_2) pc += c`             |
| BNE256_RV32  | `a,b,c,1,2` | `if([r32{0}(a):32]_2 != [r32{0}(b):32]_2) pc += c`             |
| BLT256_RV32  | `a,b,c,1,2` | `if(i256([r32{0}(a):32]_2) < i256([r32{0}(b):32]_2)) pc += c`  |
| BGE256_RV32  | `a,b,c,1,2` | `if(i256([r32{0}(a):32]_2) >= i256([r32{0}(b):32]_2)) pc += c` |
| BLTU256_RV32 | `a,b,c,1,2` | `if(u256([r32{0}(a):32]_2) < u256([r32{0}(b):32]_2)) pc += c`  |
| BGEU256_RV32 | `a,b,c,1,2` | `if(u256([r32{0}(a):32]_2) >= u256([r32{0}(b):32]_2)) pc += c` |

#### 256-bit Multiplication

Multiplication performs 256-bit×256-bit multiplication and writes the lower 256-bits to memory.
Below `x[n:m]` denotes the bits from `n` to `m` inclusive of `x`.

| Name        | Operands    | Description                                                       |
| ----------- | ----------- | ----------------------------------------------------------------- |
| MUL256_RV32 | `a,b,c,1,2` | `[r32{0}(a):32]_2 = ([r32{0}(b):32]_2 * [r32{0}(c):32]_2)[0:255]` |

### Modular Arithmetic

The VM can be configured to support intrinsic instructions for modular arithmetic. The VM configuration will specify a list of supported moduli. For each positive integer modulus `N` there will be associated configuration parameters `N::NUM_LIMBS` and `N::BLOCK_SIZE` (defined below). For each modulus `N`, the instructions below are supported.

The instructions perform operations on unsigned big integers representing elements in the modulus. The big integer values are read/written from/to memory in address space `2`. The address space `2` pointer locations are obtained by reading register values in address space `1`.

An element in the ring of integers modulo `N`is represented as an unsigned big integer with `N::NUM_LIMBS` limbs with each limb having `LIMB_BITS = 8` bits. For each instruction, the input elements `[r32{0}(b): N::NUM_LIMBS]_2, [r32{0}(c):N::NUM_LIMBS]_2` are assumed to be unsigned big integers in little-endian format with each limb having `LIMB_BITS` bits. However, the big integers are **not** required to be less than `N`. Under these conditions, the output element `[r32{0}(a): N::NUM_LIMBS]_2` written to memory will be an unsigned big integer of the same format that is congruent modulo `N` to the respective operation applied to the two inputs.

For each instruction, the operand `d` is fixed to be `1` and `e` is fixed to be `2`.
Each instruction performs block accesses with block size `4` in address space `1` and block size `N::BLOCK_SIZE` in address space `2`, where `N::NUM_LIMBS` is divisible by `N::BLOCK_SIZE`. Recall that `N::BLOCK_SIZE` must be a power of 2.

| Name             | Operands    | Description                                                                                                                                                          |
| ---------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ADDMOD_RV32\<N\> | `a,b,c,1,2` | `[r32{0}(a): N::NUM_LIMBS]_2 = [r32{0}(b): N::NUM_LIMBS]_2 + [r32{0}(c): N::NUM_LIMBS]_2 (mod N)`                                                                    |
| SUBMOD_RV32\<N\> | `a,b,c,1,2` | `[r32{0}(a): N::NUM_LIMBS]_2 = [r32{0}(b): N::NUM_LIMBS]_2 - [r32{0}(c): N::NUM_LIMBS]_2 (mod N)`                                                                    |
| MULMOD_RV32\<N\> | `a,b,c,1,2` | `[r32{0}(a): N::NUM_LIMBS]_2 = [r32{0}(b): N::NUM_LIMBS]_2 * [r32{0}(c): N::NUM_LIMBS]_2 (mod N)`                                                                    |
| DIVMOD_RV32\<N\> | `a,b,c,1,2` | `[r32{0}(a): N::NUM_LIMBS]_2 = [r32{0}(b): N::NUM_LIMBS]_2 / [r32{0}(c): N::NUM_LIMBS]_2 (mod N)`. Undefined behavior if `gcd([r32{0}(c): N::NUM_LIMBS]_2, N) != 1`. |

### Modular Branching

The configuration of `N` is the same as above. For each instruction, the input elements `[r32{0}(b): N::NUM_LIMBS]_2, [r32{0}(c): N::NUM_LIMBS]_2` are assumed to be unsigned big integers in little-endian format with each limb having `LIMB_BITS` bits.

| Name              | Operands    | Description                                                                                                                                                                                                                                                                                         |
| ----------------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ISEQMOD_RV32\<N\> | `a,b,c,1,2` | `[a:4]_1 = [r32{0}(b): N::NUM_LIMBS]_2 == [r32{0}(c): N::NUM_LIMBS]_2 (mod N) ? 1 : 0`. Enforces that `[r32{0}(b): N::NUM_LIMBS]_2, [r32{0}(c): N::NUM_LIMBS]_2` are less than `N` and then sets the register value of `[a:4]_1` to `1` or `0` depending on whether the two big integers are equal. |

### Short Weierstrass Elliptic Curve Arithmetic

The VM can be configured to support intrinsic instructions for elliptic curves `C` in short Weierstrass form given by equation `C: y^2 = x^3 + C::B` where `C::B` is a constant of the coordinate field. We note that the definitions of the curve arithmetic operations do not depend on `C::B`. The VM configuration will specify a list of supported curves. For each short Weierstrass curve `C` there will be associated configuration parameters `C::COORD_SIZE` and `C::BLOCK_SIZE` (defined below). For each curve `C`, the instructions below are supported.

An affine curve point `EcPoint(x, y)` is a pair of `x,y` where each element is an array of `C::COORD_SIZE` elements each with `LIMB_BITS = 8` bits. When the coordinate field `C::Fp` of `C` is prime, the format of `x,y` is guaranteed to be the same as the format used in the [modular arithmetic instructions](#modular-arithmetic). A curve point will be represented as `2 * C::COORD_SIZE` contiguous cells in memory.

We use the following notation below:

```
r32_ec_point(a) -> EcPoint {
    let x = [r32{0}(a): C::COORD_SIZE]_2;
    let y = [r32{0}(a) + C::COORD_SIZE: C::COORD_SIZE]_2;
    return EcPoint(x, y);
}
```

| Name           | Operands    | Description                                                                                                                                                                                                                                                                          |
| -------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| SW_ADD_NE\<C\> | `a,b,c,1,2` | Set `r32_ec_point(a) = r32_ec_point(b) + r32_ec_point(c)` (curve addition). Assumes that `r32_ec_point(b), r32_ec_point(c)` both lie on the curve and are not the identity point. Further assumes that `r32_ec_point(b).x, r32_ec_point(c).x` are not equal in the coordinate field. |
| SW_DOUBLE\<C\> | `a,b,_,1,2` | Set `r32_ec_point(a) = 2 * r32_ec_point(b)`. This doubles the input point. Assumes that `r32_ec_point(b)` lies on the curve and is not the identity point.                                                                                                                           |

### Optimal Ate Pairing

TODO

## Native Kernel

### Base

In the instructions below, `d,e` may be any valid **non-zero** address space. Base kernel instructions enable memory movement between address spaces.

In some instructions below, `W` is a generic parameter for the block size.

| Name       | Operands        | Description                                                                                                                                                                                                                                                                                 |
| ---------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LOAD\<W\>  | `a,b,c,d,e`     | Set `[a:W]_d = [[c]_d + b:W]_e`.                                                                                                                                                                                                                                                            |
| STORE\<W\> | `a,b,c,d,e`     | Set `[[c]_d + b:W]_e = [a:W]_d`.                                                                                                                                                                                                                                                            |
| LOAD2      | `a,b,c,d,e,f,g` | Set `[a]_d = [[c]_d + [f]_d * g + b]_e`.                                                                                                                                                                                                                                                    |
| STORE2     | `a,b,c,d,e,f,g` | Set `[[c]_d + [f]_d * g + b]_e = [a]_d`.                                                                                                                                                                                                                                                    |
| JAL        | `a,b,c,d`       | Jump to address and link: set `[a]_d = (pc + DEFAULT_PC_STEP)` and `pc = pc + b`.                                                                                                                                                                                                           |
| BEQ\<W\>   | `a,b,c,d,e`     | If `[a:W]_d == [b:W]_e`, then set `pc = pc + c`.                                                                                                                                                                                                                                            |
| BNE\<W\>   | `a,b,c,d,e`     | If `[a:W]_d != [b:W]_e`, then set `pc = pc + c`.                                                                                                                                                                                                                                            |
| SHINTW     | `_,b,c,d,e`     | Set `[[c]_d + b]_e = next element from hint stream`.                                                                                                                                                                                                                                        |
| PUBLISH    | `a,b,_,d,e`     | Constrains the public value at index `[a]_d` to equal `[b]_e`. Both `d,e` cannot be zero.                                                                                                                                                                                                   |
| CASTF      | `a,b,_,d,e`     | Cast a field element represented as `u32` into four bytes in little-endian: Set `[a:4]_d` to the unique array such that `sum_{i=0}^3 [a + i]_d * 2^{8i} = [b]_e` where `[a + i]_d < 2^8` for `i = 0..2` and `[a + 3]_d < 2^6`. This opcode constrains that `[b]_e` must be at most 30-bits. |

<!--
A note on CASTF: support for casting arbitrary field elements can also be supported by doing a big int less than between the block and the byte decomposition of `p`, but this seemed unnecessary and complicates the constraints.
-->

### Native Field Arithmetic

This instruction set does native field operations. Below, `d,e` may be any valid address space, and `d` is not
allowed to be zero while `e` may be zero. When `e` is zero, `[c]_0` should be interpreted as `c`.

| Name | Operands    | Description                                               |
| ---- | ----------- | --------------------------------------------------------- |
| ADDF | `a,b,c,d,e` | Set `[a]_d = [b]_d + [c]_e`.                              |
| SUBF | `a,b,c,d,e` | Set `[a]_d = [b]_d - [c]_e`.                              |
| MULF | `a,b,c,d,e` | Set `[a]_d = [b]_d * [c]_e`.                              |
| DIVF | `a,b,c,d,e` | Set `[a]_d = [b]_d / [c]_e`. Division by zero is invalid. |

### Native Extension Field Arithmetic

#### BabyBear Quartic Extension Field

This is only enabled when the native field is `BabyBear`. The quartic extension field is defined by the irreducible polynomial $x^4 - 11$ (this choice matches Plonky3, but we note that Risc0 uses the polynomial $x^4 + 11$ instead).
All elements in the field extension can be represented as a vector `[a_0,a_1,a_2,a_3]` which represents the
polynomial $a_0 + a_1x + a_2x^2 + a_3x^3$ over `BabyBear`.

Below, `d,e` may be any valid address space, and `d,e` are both not allowed to be zero. The instructions do block access with block size `4`.

| Name    | Operands  | Description                                                                                   |
| ------- | --------- | --------------------------------------------------------------------------------------------- |
| FE4ADD  | `a, b, c` | Set `[a:4]_d = [b:4]_d + [c:4]_e` with vector addition.                                       |
| FE4SUB  | `a, b, c` | Set `[a:4]_d = [b:4]_d - [c:4]_e` with vector subtraction.                                    |
| BBE4MUL | `a, b, c` | Set `[a:4]_d = [b:4]_d * [c:4]_e` with extension field multiplication.                        |
| BBE4DIV | `a, b, c` | Set `[a:4]_d = [b:4]_d / [c:4]_e` with extension field division. Division by zero is invalid. |

### Hashes

We have special opcodes to enable different precompiled hash functions.
Only subsets of these opcodes will be turned on depending on the VM use case.

Below, `d,e` may be any valid address space, and `d,e` are both not allowed to be zero. The instructions do block access with block size `1` in address space `d` and block size `CHUNK` in address space `e`.

| Name                                                                                                                                                                                                                               | Operands    | Description                                                                                                                                                                                                                                                                                                                                                      |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **COMPRESS_POSEIDON2** `[CHUNK, PID]` <br/><br/> Here `CHUNK` and `PID` are **constants** that determine different opcodes. `PID` is an internal identifier for particular Poseidon2 constants dependent on the field (see below). | `a,b,c,d,e` | Applies the Poseidon2 compression function to the inputs `[[b]_d:CHUNK]_e` and `[[c]_d:CHUNK]_e`, writing the result to `[[a]_d:CHUNK]_e`.                                                                                                                                                                                                                       |
| **PERM_POSEIDON2** `[WIDTH, PID]`                                                                                                                                                                                                  | `a,b,_,d,e` | Applies the Poseidon2 permutation function to `[[b]_d:WIDTH]_e` and writes the result to `[[a]_d:WIDTH]_e`. <br/><br/> Each array of `WIDTH` elements is read/written in two batches of size `CHUNK`. This is nearly the same as `COMPRESS_POSEIDON2` except that the whole input state is contiguous in memory, and the full output state is written to memory. |

For Poseidon2, the `PID` is just some identifier to provide domain separation between different Poseidon2 constants. For
now we can set:

| `PID` | Description                                                                                                                                                                                                                                                         |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0     | [`POSEIDON2_BABYBEAR_16_PARAMS`](https://github.com/HorizenLabs/poseidon2/blob/bb476b9ca38198cf5092487283c8b8c5d4317c4e/plain_implementations/src/poseidon2/poseidon2_instance_babybear.rs#L2023C20-L2023C48) but the Mat4 used is Plonky3's with a Monty reduction |

and only support `CHUNK = 8` and `WIDTH = 16` in BabyBear Poseidon2 above. For this setting, the input (of size `WIDTH`)
is read in two batches of size `CHUNK`, and, similarly, the output is written in either one or two batches of
size `CHUNK`, depending on the output size of the corresponding opcode.

## Phantom Sub-Instructions

As mentioned in [System](#system), the **PHANTOM** instruction has different behavior based on the operand `c`.
More specifically, the low 16-bits `c.as_canonical_u32() & 0xffff` are used as the discriminant to determine a phantom sub-instruction. We list the phantom sub-instructions below. Phantom sub-instructions are only allowed to use operands `a,b` and `c_upper = c.as_canonical_u32() >> 16`. Besides the description below, recall that the phantom instruction always advances the program counter by `DEFAULT_PC_STEP`.

| Name                     | Discriminant | Operands      | Description                                                                                                                                                           |
| ------------------------ | ------------ | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| NOP                      | 0            | `_`           | Does nothing.                                                                                                                                                         |
| DEBUG_PANIC              | 1            | `_`           | Causes the runtime to panic on the host machine and prints a backtrace if `RUST_BACKTRACE=1` is set.                                                                  |
| PRINT_F                  | 2            | `a,_,c_upper` | Prints `[a]_{c_upper}` to stdout on the host machine.                                                                                                                 |
| HINT_INPUT               | 3            | `_`           | Pops a vector `hint` of field elements from the input stream and resets the hint stream to equal the vector `[[F::from_canonical_usize(hint.len())], hint].concat()`. |
| HINT_BITS                | 4            | `a,b,c_upper` | Resets the hint stream to be the least significant `b` bits of `([a]_{c_upper}).as_canonical_u32()`.                                                                  |
| CT_START                 | 5            | `_`           | Opens a new span for tracing.                                                                                                                                         |
| CT_END                   | 6            | `_`           | Closes the current span.                                                                                                                                              |
| HINT_FINAL_EXP_BN254     | 7            | todo          | todo                                                                                                                                                                  |
| HINT_FINAL_EXP_BLS12_381 | 8            | todo          | todo                                                                                                                                                                  |

### Notes about hints

The `input_stream` is a non-interactive queue of vectors of field elements which is provided at the start of
runtime execution. The `hint_stream` is a queue of values that can be written to memory by calling `SHINTW` (which is not a phantom sub-instruction). The `hint_stream` is populated via phantom sub-instructions such
as `HINT_INPUT` (resets `hint_stream` to be the next vector popped from `input_stream`),
`HINT_BITS` (resets `hint_stream` to be the bit decomposition of a given variable, with a length known at compile time).

# RISC-V Custom Instructions

axVM intrinsic opcodes are callable from a RISC-V ELF as custom RISC-V instructions.
For these instructions, we will use the standard 32-bit RISC-V encoding unless otherwise specified.
We follow Chapter 21 of the [RISC-V spec v2.2](https://riscv.org/wp-content/uploads/2017/05/riscv-spec-v2.2.pdf) on how to extend the ISA, aiming to avoid collisions with existing standard instruction formats. As suggested by Chapter 19, for instructions which fit into 32-bits, we will use the _custom-0_ opcode[6:0] prefix **0001011** and _custom-1_ opcode[6:0] prefix **0101011**. Note that instructions are parsed from right to left, and opcode[1:0] = 11 is typical for standard 32-bit instructions (opcode[1:0]=00 is used for compressed 16-bit instructions). We will use _custom-0_ for intrinsics that don’t require additional configuration parameters, and _custom-1_ for ones that do (e.g., prime field arithmetic and elliptic curve arithmetic).

Almost all intrinsics will be R-type or I-type.

R-type format: `[funct7] [rs2] [rs1] [funct3] [rd] [opcode]`.
I-type format: `[imm[11:0]] [rs1] [funct3] [rd] [opcode]`.

We will use funct3 as the top level distinguisher between opcode classes, and then funct7 (if R-type) or imm (if I-type) for more specific specification. In the tables below, the funct7 column will specify the value of imm[11:0] when the instruction is I-type.

We start with the instructions using _custom-0_ opcode[6:0] prefix **0001011**..

## System

| RISC-V Inst | FMT | opcode[6:0] | funct3 | imm[0:11] | RISC-V description and notes                                                                |
| ----------- | --- | ----------- | ------ | --------- | ------------------------------------------------------------------------------------------- |
| terminate   | I   | 0001011     | 000    | `code`    | terminate with exit code `code`                                                             |
| hintstorew  | I   | 0001011     | 001    |           | Stores next 4-byte word from hint stream in user memory at `[rd + imm]_2` (`i32` addition). |
| reveal      | I   | 0001011     | 010    |           | Stores the 4-byte word `rs1` at address `rd + imm` in user IO space.                        |
| hintinput   | I   | 0001011     | 011    | 0x0       | Pop next vector from input stream and reset hint stream to the vector.                      |

## Hashes

| RISC-V Inst | FMT | opcode[6:0] | funct3 | funct7 | RISC-V description and notes                |
| ----------- | --- | ----------- | ------ | ------ | ------------------------------------------- |
| keccak256   | R   | 0001011     | 100    | 0x0    | `[rd:32]_2 = keccak256([rs1..rs1 + rs2]_2)` |

## 256-bit Integers

| RISC-V Inst | FMT | opcode[6:0] | funct3 | funct7 | RISC-V description and notes                              |
| ----------- | --- | ----------- | ------ | ------ | --------------------------------------------------------- |
| add256      | R   | 0001011     | 101    | 0x00   | `[rd:32]_2 = [rs1:32]_2 + [rs2:32]_2`                     |
| sub256      | R   | 0001011     | 101    | 0x01   | `[rd:32]_2 = [rs1:32]_2 - [rs2:32]_2`                     |
| xor256      | R   | 0001011     | 101    | 0x02   | `[rd:32]_2 = [rs1:32]_2 ^ [rs2:32]_2`                     |
| or256       | R   | 0001011     | 101    | 0x03   | `[rd:32]_2 = [rs1:32]_2 \| [rs2:32]_2`                    |
| and256      | R   | 0001011     | 101    | 0x04   | `[rd:32]_2 = [rs1:32]_2 & [rs2:32]_2`                     |
| sll256      | R   | 0001011     | 101    | 0x05   | `[rd:32]_2 = [rs1:32]_2 << [rs2:32]_2`                    |
| srl256      | R   | 0001011     | 101    | 0x06   | `[rd:32]_2 = [rs1:32]_2 >> [rs2:32]_2`                    |
| sra256      | R   | 0001011     | 101    | 0x07   | `[rd:32]_2 = [rs1:32]_2 >> [rs2:32]_2` MSB extends        |
| slt256      | R   | 0001011     | 101    | 0x08   | `[rd:32]_2 = i256([rs1:32]_2) < i256([rs2:32]_2) ? 1 : 0` |
| sltu256     | R   | 0001011     | 101    | 0x09   | `[rd:32]_2 = u256([rs1:32]_2) < u256([rs2:32]_2) ? 1 : 0` |
| beq256      | R   | 0001011     | 101    | 0x0a   | `if([rs1:32]_2 == [rs2:32]_2) pc += imm`                  |
| bne256      | R   | 0001011     | 101    | 0x0b   | `if([rs1:32]_2 != [rs2:32]_2) pc += imm`                  |
| blt256      | R   | 0001011     | 101    | 0x0c   | `if(i256([rs1:32]_2) < i256([rs2:32]_2)) pc += imm`       |
| bge256      | R   | 0001011     | 101    | 0x0d   | `if(i256([rs1:32]_2) >= i256([rs2:32]_2)) pc += imm`      |
| bltu256     | R   | 0001011     | 101    | 0x0e   | `if(u256([rs1:32]_2) < u256([rs2:32]_2)) pc += imm`       |
| bgeu256     | R   | 0001011     | 101    | 0x0f   | `if(u256([rs1:32]_2) >= u256([rs2:32]_2)) pc += imm`      |
| mul256      | R   | 0001011     | 101    | 0x10   | `[rd:32]_2 = ([rs1:32]_2 * [rs2:32]_2)[0:255]`            |

## Modular Arithmetic

We next proceed to the instructions using _custom-1_ opcode[6:0] prefix **0101011**..

Modular arithmetic instructions depend on the modulus `N`. The instruction set and VM can be simultaneously configured _ahead of time_ to support a fixed ordered list of moduli. We use `config.mod_idx(N)` to denote the index of `N` in this list. In the list below, `idx` denotes `config.mod_idx(N)`.

**Note:** The output for the first 4 instructions is not guaranteed to be less than `N`. See [above](#modular-arithmetic) for more details.

| RISC-V Inst  | FMT | opcode[6:0] | funct3 | funct7    | RISC-V description and notes                                                                                                                                                                                          |
| ------------ | --- | ----------- | ------ | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| addmod\<N\>  | R   | 0101011     | 000    | `idx*8`   | `[rd: N::NUM_LIMBS]_2 = [rs1: N::NUM_LIMBS]_2 + [rs2: N::NUM_LIMBS]_2 (mod N)`                                                                                                                                        |
| submod\<N\>  | R   | 0101011     | 000    | `idx*8+1` | `[rd: N::NUM_LIMBS]_2 = [rs1: N::NUM_LIMBS]_2 - [rs2: N::NUM_LIMBS]_2 (mod N)`                                                                                                                                        |
| mulmod\<N\>  | R   | 0101011     | 000    | `idx*8+2` | `[rd: N::NUM_LIMBS]_2 = [rs1: N::NUM_LIMBS]_2 * [rs2: N::NUM_LIMBS]_2 (mod N)`                                                                                                                                        |
| divmod\<N\>  | R   | 0101011     | 000    | `idx*8+3` | `[rd: N::NUM_LIMBS]_2 = [rs1: N::NUM_LIMBS]_2 / [rs2: N::NUM_LIMBS]_2 (mod N)` (undefined when `gcd([rs2: N::NUM_LIMBS]_2, N) != 1`)                                                                                  |
| iseqmod\<N\> | R   | 0101011     | 000    | `idx*8+4` | `rd = [rs1: N::NUM_LIMBS]_2 == [rs2: N::NUM_LIMBS]_2 (mod N) ? 1 : 0`. Enforces that `[rs1: N::NUM_LIMBS]_2` and `[rs2: N::NUM_LIMBS]_2` are both less than `N` and then sets `rd` equal to boolean comparison value. |

Since `funct7` is 7-bits, up to 16 moduli can be supported simultaneously. We use `idx*8` to leave some room for future expansion.

## Short Weierstrass Elliptic Curve Arithmetic

Short Weierstrass elliptic curve arithmetic depends on elliptic curve `C`. The instruction set and VM can be simultaneously configured _ahead of time_ to support a fixed ordered list of supported curves. We use `config.curve_idx(C)` to denote the index of `C` in this list. In the list below, `idx` denotes `config.curve_idx(C)`.

| RISC-V Inst    | FMT | opcode[6:0] | funct3 | funct7    | RISC-V description and notes                                                                                                                                                                  |
| -------------- | --- | ----------- | ------ | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| sw_add_ne\<C\> | R   | 0101011     | 001    | `idx*8`   | `EcPoint([rd:2*C::COORD_SIZE]_2) = EcPoint([rs1:2*C::COORD_SIZE]_2) + EcPoint([rs2:2*C::COORD_SIZE]_2)`. Assumes that input affine points are not identity and do not have same x-coordinate. |
| sw_double\<C\> | R   | 0101011     | 001    | `idx*8+1` | `EcPoint([rd:2*C::COORD_SIZE]_2) = 2 * EcPoint([rs1:2*C::COORD_SIZE]_2)`. Assumes that input affine point is not identity. `rs2` is unused and must be set to `x0`.                           |

Since `funct7` is 7-bits, up to 16 curves can be supported simultaneously. We use `idx*8` to leave some room for future expansion.

## Optimal Ate Pairing

TODO

# RISC-V to axVM Transpilation

We describe the transpilation of the RV32IM instruction set and our [custom RISC-V instructions](#risc-v-custom-instructions) to the axVM instruction set.

We use the following notation for the transpilation:

- Let `ind(rd)` denote the register index, which is in `0..32`. In particular, it fits in one field element.
- We use `itof` for the function that takes 12-bits (or 21-bits in case of J-type) to a signed integer and then mapping to the corresponding field element. So `0b11…11` goes to `-1` in `F`.
- We use `sign_extend_24` to convert a 12-bit integer into a 24-bit integer via sign extension. We use this in conjunction with `utof`, which converts 24 bits into an unsigned integer and then maps it to the corresponding field element. Note that each 24-bit unsigned integer fits in one field element.
- We use `sign_extend_16` for the analogous conversion into a 16-bit integer via sign extension.
- We use `zero_extend_24` to convert an unsigned integer with at most 24 bits into a 24-bit unsigned integer by zero extension. This is used in conjunction with `utof` to convert unsigned integers to field elements.
- The notation `imm[0:4]` means the lowest 5 bits of the immediate.

The transpilation will only be valid for programs where:

- The program code does not have program address greater than or equal to `2^PC_BITS`.
- The program does not access memory outside the range `[0, 2^addr_max_bits)`.

## RV32IM Transpilation

| RISC-V Inst | axVM Instruction                                                           |
| ----------- | -------------------------------------------------------------------------- |
| add         | ADD_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1`                               |
| sub         | SUB_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1`                               |
| xor         | XOR_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1`                               |
| or          | OR_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1`                                |
| and         | AND_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1`                               |
| sll         | SLL_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1`                               |
| srl         | SRL_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1`                               |
| sra         | SRA_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1`                               |
| slt         | SLT_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1`                               |
| sltu        | SLTU_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1`                              |
| addi        | ADD_RV32 `ind(rd), ind(rs1), utof(sign_extend_24(imm)), 1, 0`              |
| xori        | XOR_RV32 `ind(rd), ind(rs1), utof(sign_extend_24(imm)), 1, 0`              |
| ori         | OR_RV32 `ind(rd), ind(rs1), utof(sign_extend_24(imm)), 1, 0`               |
| andi        | AND_RV32 `ind(rd), ind(rs1), utof(sign_extend_24(imm)), 1, 0`              |
| slli        | SLL_RV32 `ind(rd), ind(rs1), utof(zero_extend_24(imm[0:4])), 1, 0`         |
| srli        | SRL_RV32 `ind(rd), ind(rs1), utof(zero_extend_24(imm[0:4])), 1, 0`         |
| srai        | SRA_RV32 `ind(rd), ind(rs1), utof(zero_extend_24(imm[0:4])), 1, 0`         |
| slti        | SLT_RV32 `ind(rd), ind(rs1), utof(sign_extend_24(imm)), 1, 0`              |
| sltiu       | SLTU_RV32 `ind(rd), ind(rs1), utof(sign_extend_24(imm)), 1, 0`             |
| lb          | LOADB_RV32 `ind(rd), ind(rs1), utof(sign_extend_16(imm)), 1, 2`            |
| lh          | LOADH_RV32 `ind(rd), ind(rs1), utof(sign_extend_16(imm)), 1, 2`            |
| lw          | LOADW_RV32 `ind(rd), ind(rs1), utof(sign_extend_16(imm)), 1, 2`            |
| lbu         | LOADBU_RV32 `ind(rd), ind(rs1), utof(sign_extend_16(imm)), 1, 2`           |
| lhu         | LOADHU_RV32 `ind(rd), ind(rs1), utof(sign_extend_16(imm)), 1, 2`           |
| sb          | STOREB_RV32 `ind(rs2), ind(rs1), utof(sign_extend_16(imm)), 1, 2`          |
| sh          | STOREH_RV32 `ind(rs2), ind(rs1), utof(sign_extend_16(imm)), 1, 2`          |
| sw          | STOREW_RV32 `ind(rs2), ind(rs1), utof(sign_extend_16(imm)), 1, 2`          |
| beq         | BEQ_RV32 `ind(rs1), ind(rs2), itof(imm), 1, 1`                             |
| bne         | BNE_RV32 `ind(rs1), ind(rs2), itof(imm), 1, 1`                             |
| blt         | BLT_RV32 `ind(rs1), ind(rs2), itof(imm), 1, 1`                             |
| bge         | BGE_RV32 `ind(rs1), ind(rs2), itof(imm), 1, 1`                             |
| bltu        | BLTU_RV32 `ind(rs1), ind(rs2), itof(imm), 1, 1`                            |
| bgeu        | BGEU_RV32 `ind(rs1), ind(rs2), itof(imm), 1, 1`                            |
| jal         | JAL_RV32 `ind(rd), 0, itof(imm), 1, 0, (rd != x0)`                         |
| jalr        | JALR_RV32 `ind(rd), ind(rs1), utof(sign_extend_16(imm)), 1, 0, (rd != x0)` |
| lui         | LUI_RV32 `ind(rd), 0, utof(zero_extend_24(imm[12:31])), 1, 0, 1`           |
| auipc       | AUIPC_RV32 `ind(rd), 0, utof(zero_extend_24(imm[12:31]) << 4), 1`          |
| mul         | MUL_RV32 `ind(rd), ind(rs1), ind(rs2), 1`                                  |
| mulh        | MULH_RV32 `ind(rd), ind(rs1), ind(rs2), 1`                                 |
| mulhsu      | MULHSU_RV32 `ind(rd), ind(rs1), ind(rs2), 1`                               |
| mulhu       | MULHU_RV32 `ind(rd), ind(rs1), ind(rs2), 1`                                |
| div         | DIV_RV32 `ind(rd), ind(rs1), ind(rs2), 1`                                  |
| divu        | DIVU_RV32 `ind(rd), ind(rs1), ind(rs2), 1`                                 |
| rem         | REM_RV32 `ind(rd), ind(rs1), ind(rs2), 1`                                  |
| remu        | REMU_RV32 `ind(rd), ind(rs1), ind(rs2), 1`                                 |

## Custom Instruction Transpilation

| RISC-V Inst    | axVM Instruction                                              |
| -------------- | ------------------------------------------------------------- |
| terminate      | TERMINATE `_, _, utof(imm)`                                   |
| hintstorew     | HINTSTOREW_RV32 `0, ind(rd), utof(sign_extend_16(imm)), 1, 2` |
| reveal         | REVEAL_RV32 `0, ind(rd), utof(sign_extend_16(imm)), 1, 3`     |
| hintinput      | PHANTOM `_, _, HintInput as u16`                              |
| keccak256      | KECCAK256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`            |
| add256         | ADD256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`               |
| sub256         | SUB256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`               |
| xor256         | XOR256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`               |
| or256          | OR256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`                |
| and256         | AND256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`               |
| sll256         | SLL256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`               |
| srl256         | SRL256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`               |
| sra256         | SRA256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`               |
| slt256         | SLT256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`               |
| sltu256        | SLTU256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`              |
| beq256         | BEQ256_RV32 `ind(rs1), ind(rs2), itof(imm), 1, 2`             |
| bne256         | BNE256_RV32 `ind(rs1), ind(rs2), itof(imm), 1, 2`             |
| blt256         | BLT256_RV32 `ind(rs1), ind(rs2), itof(imm), 1, 2`             |
| bge256         | BGE256_RV32 `ind(rs1), ind(rs2), itof(imm), 1, 2`             |
| bltu256        | BLTU256_RV32 `ind(rs1), ind(rs2), itof(imm), 1, 2`            |
| bgeu256        | BGEU256_RV32 `ind(rs1), ind(rs2), itof(imm), 1, 2`            |
| mul256         | MUL256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`               |
| addmod\<N\>    | ADDMOD_RV32\<N\> `ind(rd), ind(rs1), ind(rs2), 1, 2`          |
| submod\<N\>    | SUBMOD_RV32\<N\> `ind(rd), ind(rs1), ind(rs2), 1, 2`          |
| mulmod\<N\>    | MULMOD_RV32\<N\> `ind(rd), ind(rs1), ind(rs2), 1, 2`          |
| divmod\<N\>    | DIVMOD_RV32\<N\> `ind(rd), ind(rs1), ind(rs2), 1, 2`          |
| iseqmod\<N\>   | ISEQMOD_RV32\<N\> `ind(rd), ind(rs1), ind(rs2), 1, 2`         |
| sw_add_ne\<C\> | SW_ADD_NE_RV32\<C\> `ind(rd), ind(rs1), ind(rs2), 1, 2`       |
| sw_double\<C\> | SW_DOUBLE_RV32\<C\> `ind(rd), ind(rs1), 0, 1, 2`              |
