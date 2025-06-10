# OpenVM Instruction Set Architecture

OpenVM supports an extensible instruction set, with different groups of opcodes supported by different VM extensions.
This specification describes the overall architecture and default VM extensions which ship with OpenVM, which are:

- [RV32IM](#rv32im-extension): An extension supporting the 32-bit RISC-V ISA with multiplication.
- [Native](#native-extension): An extension supporting native field arithmetic for proof recursion and aggregation.
- [Keccak-256](#keccak-extension): An extension implementing the Keccak-256 hash function compatibly with RISC-V memory.
- [SHA2-256](#sha2-256-extension): An extension implementing the SHA2-256 hash function compatibly with RISC-V memory.
- [BigInt](#bigint-extension): An extension supporting 256-bit signed and unsigned integer arithmetic, including
  multiplication. This extension respects the RISC-V memory format.
- [Algebra](#algebra-extension): An extension supporting modular arithmetic over arbitrary fields and their complex
  field extensions. This extension respects the RISC-V memory format.
- [Elliptic curve](#elliptic-curve-extension): An extension for elliptic curve operations over Weierstrass curves,
  including addition and doubling. This can be used to implement multi-scalar multiplication and ECDSA scalar
  multiplication. This extension respects the RISC-V memory format.
- [Pairing](#pairing-extension): An extension containing opcodes used to implement the optimal Ate pairing on the BN254
  and BLS12-381 curves. This extension respects the RISC-V memory format.

In addition to these default extensions, developers are able to extend the ISA by defining their own custom VM
extensions. For reader convenience, we provide a mapping between the code-level representation of opcodes in OpenVM and
the opcodes below [here](./isa-table.md).

## Constants and Configuration Parameters

OpenVM depends on the following parameters, some of which are fixed and some of which are configurable:

| Name                | Description                                                        | Constraints                                                                                         |
| ------------------- | ------------------------------------------------------------------ |-----------------------------------------------------------------------------------------------------|
| `F`                 | The field over which the VM operates.                              | Currently fixed to Baby Bear, but may change to another 31-bit field.                               |
| `PC_BITS`           | The number of bits in the program counter.                         | Fixed to 30.                                                                                        |
| `DEFAULT_PC_STEP`   | The default program counter step size.                             | Fixed to 4.                                                                                         |
| `LIMB_BITS`         | The number of bits in a limb for RISC-V memory emulation.          | Fixed to 8.                                                                                         |
| `as_offset`         | The index of the first writable address space.                     | Fixed to 1.                                                                                         |
| `as_height`         | The base 2 log of the number of writable address spaces supported. | Configurable, must satisfy `as_height <= F::bits() - 2`                                             |
| `pointer_max_bits`  | The maximum number of bits in a pointer.                           | Configurable, must satisfy `pointer_max_bits <= F::bits() - 2`                                      |
| `num_public_values` | The number of user public values.                                  | Configurable. If continuation is enabled, it must equal `8` times a power of two(which is nonzero). |

We explain these parameters in subsequent sections.

### Prime Field

The ISA globally depends on a prime field `F`. This field is currently fixed to Baby Bear, with modulus `15 * 2^27 + 1`, but it may change in the future to another
31-bit field. Unless otherwise specified, we will identify the elements of `F` with the integers `{0, ..., F::modulus() - 1}`,
where `F::modulus()` is the prime modulus of the field.

## Virtual Machine State

The virtual machine is a state machine that is executed on a physical host machine.
The state of the virtual machine consists of the following components:

**Guest State**:

- [Program ROM](#program-rom) (Read Only)
- [Program Counter](#program-counter) `pc`
- [Data Memory](#data-memory) (Read/Write)
- [User Public Outputs](#user-public-outputs)

**Host State**:

- [Input Stream](#inputs-and-hints)
- [Hint Stream](#inputs-and-hints)
- [Hint Spaces](#inputs-and-hints)

The **initial state** of the virtual machine consists of:

- Program ROM - immutable throughout VM execution
- `pc_0` - starting program counter
- Initial data memory
- No user public outputs
- Input stream
- Empty hint stream
- Empty hint spaces

We describe these components in more detail below.

### Program ROM

OpenVM operates under the Harvard architecture, where program code is stored separately from data
memory. The program code is loaded as read-only memory (ROM) in the VM state prior to execution, and it remains
immutable throughout the execution.

Program code is a map from `[0, 2^PC_BITS)` to the space of instructions `F^{NUM_OPERANDS + 1}`, where

- `PC_BITS = 30`
- `NUM_OPERANDS = 7`.

Instructions will typically only exist at a subset of the indices in `[0, 2^PC_BITS)`.

#### Instruction format

Instructions are encoded as a global opcode (field element) followed by `NUM_OPERANDS = 7` operands (field elements):

```
opcode, a, b, c, d, e, f, g
```

An instruction does not need to use all operands, and trailing unused operands are suggested to be
set to zero, but this won't be checked.

In the following sections, you will see operands like `a, b, c, 1, e`. `1` here means a fixed address space. In this
case, `d` is suggested be set to `1`, but this won't be checked.

### Program Counter

There is a single special purpose register `pc` for the program counter of type `F` which stores the location of the
instruction being executed. During execution, the program counter must always be a valid program address, meaning that it
is an element in the range `[0, 2^PC_BITS)` where the program code is defined.

### Data Memory

Data memory is a random access memory (RAM) which supports read and write operations. Memory is comprised of addressable
cells which represent a single field element indexed by **address space** and **pointer**. The number of supported
address spaces and the size of each address space are configurable constants.

- Valid address spaces not used for immediates lie in `[1, 1 + 2^as_height)` for configuration constant `as_height`.
- Valid pointers are field elements that lie in `[0, 2^pointer_max_bits)`, for
  configuration constant `pointer_max_bits`. When accessing an address out of `[0, 2^pointer_max_bits)`, the VM should
  panic.

These configuration constants must satisfy `as_height, pointer_max_bits <= F::bits() - 2`. We use the following notation
to denote cells in memory:

- `[a]_d` denotes the single-cell value at pointer location `a` in address space `d`. This is a single
  field element.
- `[a:N]_d` denotes the slice `[a..a + N]_d` -- this is a length-`N` array of field elements.

#### Immediates

We reserve a special address space `0` for immediates. The framework enforces that address space `0` is never written
to. Address space `0` is considered a read-only array with `[a]_0 = a` for any `a` in `F`.

#### Memory Accesses and Block Accesses

VM instructions can access (read or write) a contiguous list of cells (called a **block**) in a single address space.
The block size must be in the set `{1, 2, 4, 8, 16, 32}`, and the access does not need to be aligned, meaning that
it can start from any pointer address, even those not divisible by the block size. An access is called a **block access
** if it has size greater than 1. Block accesses are not supported for address space `0`.

#### Address Spaces

Different address spaces are used for different purposes in OpenVM. Memory cells in all address spaces are always field
elements, but certain address spaces may impose the additional constraint that all elements fit into a maximum number of
bits. The existing extensions reference the following set of address spaces, but user-defined extensions are free to
introduce additional address spaces:

| Address Space | Name        | Notes and Constraints                                                             |
| ------------- | ----------- | --------------------------------------------------------------------------------- |
| `0`           | Immediates  | Address space `0` is reserved for denoting immediates, and we define `[a]_0 = a`. |
| `1`           | Registers   | Elements are constrained to lie in `[0, 2^LIMB_BITS)` for `LIMB_BITS = 8`.        |
| `2`           | User Memory | Elements are constrained to lie in `[0, 2^LIMB_BITS)` for `LIMB_BITS = 8`.        |
| `3`           | User IO     |                                                                                   |
| `4`           | Native      | Elements are typically full native field elements.                                |

When adding a new user address space, the invariants of the memory cells in that address space must be declared, and all
instructions must ensure that these invariants are preserved.

ℹ️ When adding a new instruction to the ISA, the instruction **must declare its supported address spaces** and respect
the invariants of those address spaces. In particular, all instructions must respect the invariants of the address spaces above.

### Inputs and Hints

To enable user input and non-determinism in OpenVM programs, the host state maintains the following three data
structures during runtime execution:

- `input_stream`: a private non-interactive queue of vectors of field elements which is provided at the start of runtime
  execution
- `hint_stream`: a queue of values populated during runtime execution
  via [phantom sub-instructions](#phantom-sub-instructions) such as `Rv32HintInput`, `NativeHintInput`, and
  `NativeHintBits`.
- `hint_space`: a vector of vectors of field elements used to store hints during runtime execution
  via [phantom sub-instructions](#phantom-sub-instructions) such as `NativeHintLoad`. The outer `hint_space` vector is append-only, but
  each internal `hint_space[hint_id]` vector may be mutated, including deletions, by the host.
- `kv_store`: a read-only key-value store for hints. Executors(e.g. `Rv32HintLoadByKey`) can read data from `kv_store` 
  at runtime. `kv_store` is designed for general purposes so both key and value are byte arrays. Encoding of key/value
  are decided by each executor. Users need to use the corresponding encoding when adding data to `kv_store`.

These data structures are **not** part of the guest state, and their state depends on host behavior that cannot be determined by the guest.

### User Public Outputs

To make program outputs public, OpenVM allows the user to specify a list of field elements to make public. To populate
this list, users can use:

- If continuations are enabled: `REVEAL_RV32` (from the RV32IM extension)
- If continuations are disabled: `PUBLISH` (from the system extension)

The list is of length `num_public_values`, where `num_public_values` is a VM configuration parameter. By default, any element in the list that is never initialized is set to zero.

## Instruction Execution

Starting from the initial VM state, the VM executes instructions according to the program ROM. Instruction execution is
a transition function on the mutable parts of the VM state:

- Program Counter
- Data Memory
- User Public Outputs
- Input Stream
- Hint Stream
- Hint Spaces

which must satisfy the following conditions:

- Let `from_pc` be the program counter at the start of the instruction execution. The `from_pc` must be the address of a
  valid instruction in the program ROM.
- The execution must match the instruction from the program ROM.
- The execution has full read/write access to the data memory, except address space `0` must be read-only.
- User public outputs can be set at any index in `[0, num_public_values)`. If continuations are disabled, a public
  value cannot be overwritten with a different value once it is set.
- Input stream can only be popped from the front as a queue.
- Full read/write access to the hint stream.
- Hint spaces can be read from at any index. Hint spaces may be mutated only by append.
- The program counter is set to a new `to_pc` at the end of the instruction execution.
  Instructions are only considered valid if `to_pc` is the address of a valid instruction in the program ROM.

ℹ️ Notes for extension developers: The specification of new instructions should carefully consider `to_pc` overflows, especially when you want to move `pc` with
a positive offset.

### Guest Instruction Execution

We define **guest instruction execution** to be the subset of instruction execution that only mutates the guest state:

- Program Counter
- Data Memory
- User Public Outputs

Guest instruction execution may still depend on read access to the host state.
For example, instructions like `HINT_STORE_RV32` (from the RV32IM extension) and `HINT_STOREW`, `HINT_STOREW4` (from the native extension) can
read from the `hint_stream` and write them to OpenVM memory to provide non-deterministic hints.

⚠️ Safeguards:

- All instructions should ensure that the modifications to the guest state are protected from non-deterministic host states, as the guest has no control
  over the host state. For example,
  the start address and length of modified guest memory must be derived from instruction operands or guest state (as opposed to being derived from host state).

### Phantom Instructions

To facilitate hinting and debugging on the host, OpenVM supports the notion of **phantom instructions**. These are
instructions which are identical to a no-op at the level of the OpenVM guest state, but which may be used to specify
unconstrained behavior on the host. Use cases of phantom instructions include interacting with the input or hint streams
or displaying debug information on the host machine.

Notes for extension developers: `PhantomDiscriminant` should be unique for each phantom instruction. If you want your
new extension to be compatible with some extensions, you need select some compatible `PhantomDiscriminant`.

## OpenVM Instruction Set

We now specify instructions supported by the default VM extensions shipping with OpenVM. Unless otherwise specified,
instructions will set `to_pc = from_pc + DEFAULT_PC_STEP`. We will use the following notation:

- `u32(x)` where `x: [F; 4]` consists of 4 bytes will mean the casting from little-endian bytes in 2's complement to
  unsigned 32-bit integer.
- `i32(x)` where `x: [F; 4]` consists of 4 bytes will mean the casting from little-endian bytes in 2's complement to
  signed 32-bit integer.
- `sign_extend` means sign extension of bits in 2's complement.
- `i32(c)` where `c` is a field element will mean `c.as_canonical_u32()` if `c.as_canonical_u32() < F::modulus() / 2` or
  `c.as_canonical_u32() - F::modulus() as i32` otherwise.
- `decompose(c)` where `c` is a field element means `c.as_canonical_u32().to_le_bytes()`.
- `r32{0}(a) := i32([a:4]_1)` means casting the value at `[a:4]_1` to `i32`.

In the specification, operands marked with `_` are not used and should be set to zero. Trailing unused operands should
also be set to zero.

### System Instructions

The opcodes below are supported by the OpenVM system and do not belong to any VM extension.

| Name      | Operands    | Description                                                                                                                                                                   |
| --------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| TERMINATE | `_, _, c`   | Terminates execution with exit code `c`. Sets `to_pc = from_pc`.                                                                                                              |
| PHANTOM   | `_, _, c`   | Sets `to_pc = from_pc + DEFAULT_PC_STEP`. The operand `c` determines which phantom instruction (see below) is run.                                                            |
| PUBLISH   | `a,b,_,d,e` | Set the user public output at index `[a]_d` to equal `[b]_e`. Invalid if `[a]_d` is greater than or equal to `num_public_values`. Only valid when continuations are disabled. |

The behavior of the PHANTOM opcode is determined by the operand `c`.
More specifically, the low 16-bits `c.as_canonical_u32() & 0xffff` are used as a discriminant to determine a phantom
sub-instruction. Phantom sub-instructions supported by the system are listed below, and VM extensions can define
additional phantom sub-instructions.
Phantom sub-instructions are only allowed to use operands `a,b` and `c_upper = c.as_canonical_u32() >> 16` and must
always advance the program counter by `DEFAULT_PC_STEP`.

| Name       | Discriminant | Operands | Description                                                                                          |
| ---------- | ------------ | -------- | ---------------------------------------------------------------------------------------------------- |
| Nop        | 0x00         | `_`      | Does nothing.                                                                                        |
| DebugPanic | 0x01         | `_`      | Causes the runtime to panic on the host machine and prints a backtrace if `RUST_BACKTRACE=1` is set. |
| CtStart    | 0x02         | `_`      | Opens a new span for tracing.                                                                        |
| CtEnd      | 0x03         | `_`      | Closes the current span.                                                                             |

### RV32IM Extension

The RV32IM extension introduces OpenVM opcodes which support 32-bit RISC-V via transpilation from a standard RV32IM ELF
binary, specified [here](./RISCV.md). These consist of opcodes corresponding 1-1 with RV32IM opcodes, as well as
additional user IO opcodes and phantom sub-instructions to support input and debug printing on the host. We denote the
OpenVM opcode corresponding to a RV32IM opcode by appending `_RV32`.

The RV32IM extension uses address space `0` for immediates, address space `1` for registers, and address space `2` for
memory. As explained [here](#address-spaces), cells in address spaces `1` and `2` are constrained to be bytes, and all
instructions preserve this constraint.

The `i`th RISC-V register is represented by the block `[4 * i:4]_1` of 4 limbs in address space `1`. All memory
addresses in address space `1` behave uniformly, and in particular writes to the block `[0:4]_1` corresponding to the
RISC-V register `x0` are allowed in the RV32IM extension. However, as detailed
in [RV32IM Transpilation](./transpiler.md#rv32im-transpilation), any OpenVM program transpiled from a RV32IM ELF will
never contain such a write and conforms to the RV32IM specification.

#### ALU

In all ALU instructions, the operand `d` is fixed to be `1`. The operand `e` must be either `0` or `1`. When `e = 0`,
the `c` operand is expected to be of the form `F::from_canonical_u32(c_i16 as i24 as u24 as u32)` where `c_i16` is type
`i16`. In other words we take signed 16-bits in two's complement, sign extend to 24-bits, consider the 24-bits as
unsigned integer, and convert to field element. In the instructions below, `[c:4]_0` should be interpreted as
`c_i16 as i32` sign extended to 32-bits.

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

#### Load/Store

For all load/store instructions, we assume the operand `c` is in `[0, 2^16)`, and we fix address spaces `d = 1`.
The address space `e` can be `0`, `1`, or `2` for load instructions, and `2`, `3`, or `4` for store instructions.
The operand `g` must be a boolean. We let `sign_extend(decompose(c)[0:2], g)` denote the `i32` defined by first taking
the unsigned integer encoding of `c` as 16 bits, then sign extending it to 32 bits using the sign bit `g`, and considering
the 32 bits as the 2's complement of an `i32`.
We will use shorthand `r32{c,g}(b) := i32([b:4]_1) + sign_extend(decompose(c)[0:2], g)` as `i32`. This means performing
signed 32-bit addition with the value of the register `[b:4]_1`. For consistency with other notation,
we define the shorthand `r32{c}(b)` to mean `r32{c,g}(b)` where `g` is set to the most significant bit of `c`.
Memory access to `ptr: i32` in address space `e` is only valid if `0 <= ptr < 2^addr_max_bits` and
`ptr` is naturally aligned (i.e., `ptr` must be divisible by the data size in bytes), in
which case it is an access to `F::from_canonical_u32(ptr as u32)`.
The data size is `1` for LOADB_RV32, LOADBU_RV32, STOREB_RV32, `2` for LOADH_RV32, LOADHU_RV32, STOREH_RV32, and `4` for LOADW_RV32, STOREW_RV32.

All load/store instructions always do block accesses of block size `4`, even for LOADB_RV32 and STOREB_RV32.

| Name        | Operands        | Description                                                                                                                                                                         |
| ----------- | --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LOADB_RV32  | `a,b,c,1,e,f,g` | `if(f!=0) [a:4]_1 = sign_extend([r32{c,g}(b):1]_e)` The operand `f` is assumed to be boolean. Must sign-extend the byte read from memory, which is represented in 2’s complement.   |
| LOADH_RV32  | `a,b,c,1,e,f,g` | `if(f!=0) [a:4]_1 = sign_extend([r32{c,g}(b):2]_e)` The operand `f` is assumed to be boolean. Must sign-extend the number read from memory, which is represented in 2’s complement. |
| LOADW_RV32  | `a,b,c,1,e,f,g` | `if(f!=0) [a:4]_1 = [r32{c,g}(b):4]_e` The operand `f` is assumed to be boolean.                                                                                                    |
| LOADBU_RV32 | `a,b,c,1,e,f,g` | `if(f!=0) [a:4]_1 = zero_extend([r32{c,g}(b):1]_e)` The operand `f` is assumed to be boolean. Must zero-extend the number read from memory.                                         |
| LOADHU_RV32 | `a,b,c,1,e,f,g` | `if(f!=0) [a:4]_1 = zero_extend([r32{c,g}(b):2]_e)` The operand `f` is assumed to be boolean. Must zero-extend the number read from memory.                                         |
| STOREB_RV32 | `a,b,c,1,e,1,g` | `[r32{c,g}(b):1]_e <- [a:1]_1`                                                                                                                                                      |
| STOREH_RV32 | `a,b,c,1,e,1,g` | `[r32{c,g}(b):2]_e <- [a:2]_1`                                                                                                                                                      |
| STOREW_RV32 | `a,b,c,1,e,1,g` | `[r32{c,g}(b):4]_e <- [a:4]_1`                                                                                                                                                      |

#### Branch/Jump/Upper Immediate

For branch instructions, we fix `d = e = 1`. For jump instructions, we fix `d = 1`.

| Name       | Operands         | Description                                                                                                                                                                                                                                                                                                                                                                 |
| ---------- | ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| BEQ_RV32   | `a,b,c,1,1`      | `if([a:4]_1 == [b:4]_1) pc += c`                                                                                                                                                                                                                                                                                                                                            |
| BNE_RV32   | `a,b,c,1,1`      | `if([a:4]_1 != [b:4]_1) pc += c`                                                                                                                                                                                                                                                                                                                                            |
| BLT_RV32   | `a,b,c,1,1`      | `if(i32([a:4]_1) < i32([b:4]_1)) pc += c`                                                                                                                                                                                                                                                                                                                                   |
| BGE_RV32   | `a,b,c,1,1`      | `if(i32([a:4]_1) >= i32([b:4]_1)) pc += c`                                                                                                                                                                                                                                                                                                                                  |
| BLTU_RV32  | `a,b,c,1,1`      | `if(u32([a:4]_1) < u32([b:4]_1)) pc += c`                                                                                                                                                                                                                                                                                                                                   |
| BGEU_RV32  | `a,b,c,1,1`      | `if(u32([a:4]_1) >= u32([b:4]_1)) pc += c`                                                                                                                                                                                                                                                                                                                                  |
| JAL_RV32   | `a,_,c,1,_,f`    | `if(f!=0) [a:4]_1 = decompose(pc+4); pc += c`. The operand `f` is assumed to be boolean. The `pc` increment is always done regardless of `f`'s value. Here `i32(c)`must be in`[-2^24, 2^24)`.                                                                                                                                                                                 |
| JALR_RV32  | `a,b,c,1,_,f,g` | `if(f!=0) [a:4]_1 = decompose(pc+4); pc = F::from_canonical_u32(i32([b:4]_1) + sign_extend(decompose(c)[0:2], g) as u32)`. Constrains that `i32([b:4]_1) + sign_extend(decompose(c)[0:2], g) is in [0, 2^PC_BITS)`. Here `u32(c)` must be in `[0, 2^16)`. The operands `f` and `g` are assumed to be boolean. The `pc` assignment is always done regardless of `f`'s value. |
| LUI_RV32   | `a,_,c,1,_,1`    | `[a:4]_1 = u32(c) << 12`. Here `i32(c)` must be in `[0, 2^20)`.                                                                                                                                                                                                                                                                                                             |
| AUIPC_RV32 | `a,_,c,1,_,_`    | `[a:4]_1 = decompose(pc) + (decompose(c) << 8)`. Here `i32(c)` must be in `[0, 2^24)`.                                                                                                                                                                                                                                                                                     |

For branch and JAL_RV32 instructions, the instructions assume that the operand `i32(c)` is in `[-2^24,2^24)`. The
assignment `pc += c` is done as field elements.
In valid programs, the `from_pc` is always in `[0, 2^PC_BITS)`. We will only use base field `F` satisfying
`2^PC_BITS + 2*2^24 < F::modulus()` so `to_pc = from_pc + c` is only valid if `i32(from_pc) + i32(c)` is in
`[0, 2^PC_BITS)`.

For JALR_RV32, we treat `c` in `[0, 2^16)` as a raw encoding of 16-bits.
The operand `g` must be a boolean. We let `sign_extend(decompose(c)[0:2], g)` denote the `i32` defined by first taking
the unsigned integer encoding of `c` as 16 bits, then sign extending it to 32 bits using the sign bit `g`, and considering the 32 bits as the 2's complement of an `i32`. Then it is added to the register value `i32([b:4]_1)`, where 32-bit overflow is ignored. The instruction is only valid if the resulting `i32` is in range `[0, 2^PC_BITS)`. The
result is then cast to `u32` and then to `F` and assigned to `pc`.

For LUI_RV32, we are treating `c` in `[0, 2^20)` as a raw encoding of 20-bits.
For AUIPC_RV32, we are treating `c` in `[0, 2^24)` as a raw encoding of 24-bits.
The instruction does not need to interpret whether the register is signed or unsigned.
For AUIPC_RV32, the addition is treated as unchecked `u32` addition since that is the same as `i32` addition at the bit
level.

Note that AUIPC_RV32 does not have any condition for the register write.

#### RV32M Extension

For multiplication extension instructions, we fix `d = 1`.
MUL_RV32 performs an 32-bit×32-bit multiplication and places the lower 32 bits in the
destination cells. MULH_RV32, MULHU_RV32, and MULHSU_RV32 perform the same multiplication but return
the upper 32 bits of the full 2×32-bit product, for signed×signed, unsigned×unsigned, and
signed×unsigned multiplication respectively.

DIV_RV32 and DIVU_RV32 perform signed and unsigned integer division of 32-bits by 32-bits. REM_RV32
and REMU_RV32 provide the remainder of the corresponding division operation. Integer division is defined by
`dividend = q * divisor + r` where `0 <= |r| < |divisor|` and either `sign(r) = sign(dividend)` or `r = 0`.

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

#### User IO

In addition to opcodes which match 1-1 with the RV32IM opcodes, the following additional
opcodes interact with address spaces outside of 1 and 2 to enable verification of programs
with user input-output.

| Name             | Operands        | Description                                                                                                                                                                       |
| ---------------- | --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| HINT_STOREW_RV32 | `_,b,_,1,2`     | `[r32{0}(b):4]_2 = next 4 bytes from hint stream`. Only valid if next 4 values in hint stream are bytes.                                                                          |
| HINT_BUFFER_RV32 | `a,b,_,1,2`     | `[r32{0}(b):4 * l]_2 = next 4 * l bytes from hint stream` where `l = r32{0}(a)`. Only valid if next `4 * l` values in hint stream are bytes. Very important: `l` should not be 0. The pointer address `r32{0}(b)` does not need to be a multiple of `4`. |
| REVEAL_RV32      | `a,b,c,1,3,_,g` | Pseudo-instruction for `STOREW_RV32 a,b,c,1,3,_,g` writing to the user IO address space `3`. Only valid when continuations are enabled.                                           |

#### Phantom Sub-Instructions

The RV32IM extension defines the following phantom sub-instructions.

| Name              | Discriminant | Operands | Description                                                                                                                                                                                                                                                    |
|-------------------| ------------ | -------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Rv32HintInput     | 0x20         | `_`      | Pops a vector `hint` of field elements from the input stream and resets the hint stream to equal the vector `[(hint.len() as u32).to_le_bytes()), hint].concat()`.                                                                                             |
| Rv32PrintStr      | 0x21         | `a,b,_`  | Peeks at `[r32{0}(a)..r32{0}(a) + r32{0}(b)]_2`, tries to convert to byte array and then UTF-8 string and prints to host stdout. Prints error message if conversion fails. Does not change any VM state.                                                       |
| Rv32HintRandom    | 0x22         | `a,_,_`  | Resets the hint stream to `4 * r32{0}(a)` random bytes. The source of randomness is the host operating system (`rand::rngs::OsRng`). Its result is not constrained in any way.                                                                                 |
| Rv32HintLoadByKey | 0x23         | `a,b,_`  | Look up the value by key `[r32{0}{a}:r32{0}{b}]_2` and prepend the value into `input_stream`. The logical value is `Vec<Vec<F>>`. The serialization of `Vec` follows the format `[length, <content>]`. Both length and content encoded as little-endian bytes. |
### Native Extension

The native extension operates over native field elements and has instructions tailored for STARK proof recursion. It
does not constrain memory elements to be bytes and most instructions only write to address space `4`, with the notable
exception of CASTF.

#### Base

In the instructions below, `d,e` must be either `0` or `4` except in CASTF, which may write to address space `2`.
Additional restrictions are applied on a per-instruction basis. In particular, the immediate address space `0` is
allowed for non-vectorized
reads but not allowed for writes. When using immediates, we interpret `[a]_0` as the immediate value `a`.

| Name         | Operands    | Description                                                                                                                                                                                                                                                                                |
|--------------|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| LOADW        | `a,b,c,4,4` | Set `[a]_4 = [[c]_4 + b]_4`.                                                                                                                                                                                                                                                               |
| STOREW       | `a,b,c,4,4` | Set `[[c]_4 + b]_4 = [a]_4`.                                                                                                                                                                                                                                                               |
| LOADW4       | `a,b,c,4,4` | Set `[a:4]_4 = [[c]_4 + b:4]_4`.                                                                                                                                                                                                                                                           |
| STOREW4      | `a,b,c,4,4` | Set `[[c]_4 + b:4]_4 = [a:4]_4`.                                                                                                                                                                                                                                                           |
| JAL          | `a,b,_,4`   | Jump to address and link: set `[a]_4 = (pc + DEFAULT_PC_STEP)` and `pc = pc + b`.                                                                                                                                                                                                          |
| RANGE_CHECK  | `a,b,c,4`   | Assert that `[a]_4 = x + y * 2^16` for some `x < 2^b` and `y < 2^c`. `b` must be in [0,16] and `c` must be in [0, 14].                                                                                                                                                                     |
| BEQ          | `a,b,c,d,e` | If `[a]_d == [b]_e`, then set `pc = pc + c`.                                                                                                                                                                                                                                               |
| BNE          | `a,b,c,d,e` | If `[a]_d != [b]_e`, then set `pc = pc + c`.                                                                                                                                                                                                                                               |
| HINT_STOREW  | `_,b,c,4,4` | Set `[[c]_4 + b]_4 = next element from hint stream`.                                                                                                                                                                                                                                       |
| HINT_STOREW4 | `_,b,c,4,4` | Set `[[c]_4 + b:4]_4 = next 4 elements from hint stream`.                                                                                                                                                                                                                                  |
| CASTF        | `a,b,_,2,4` | Cast a field element represented as `u32` into four bytes in little-endian: Set `[a:4]_2` to the unique array such that `sum_{i=0}^3 [a + i]_2 * 2^{8i} = [b]_4` where `[a + i]_2 < 2^8` for `i = 0..2` and `[a + 3]_2 < 2^6`. This opcode constrains that `[b]_4` must be at most 30-bits. |

#### Field Arithmetic

This instruction set does native field operations. Below, `e,f` may be any address space.
When either `e` or `f` is zero, `[b]_0` and `[c]_0` should be interpreted as the immediates `b`
and `c`, respectively.

| Name | Operands      | Description                                               |
| ---- | ------------- | --------------------------------------------------------- |
| ADDF | `a,b,c,4,e,f` | Set `[a]_4 = [b]_e + [c]_f`.                              |
| SUBF | `a,b,c,4,e,f` | Set `[a]_4 = [b]_e - [c]_f`.                              |
| MULF | `a,b,c,4,e,f` | Set `[a]_4 = [b]_e * [c]_f`.                              |
| DIVF | `a,b,c,4,e,f` | Set `[a]_4 = [b]_e / [c]_f`. Division by zero is invalid. |

#### Extension Field Arithmetic

This is only enabled when the native field is `BabyBear`. The quartic extension field is defined by the irreducible
polynomial $x^4 - 11$, which matches Plonky3.
All elements in the field extension can be represented as a vector `[a_0,a_1,a_2,a_3]` which represents the
polynomial $a_0 + a_1x + a_2x^2 + a_3x^3$ over `BabyBear`.

The instructions read and write from address space `4` and do block access with block size `4`.

| Name    | Operands        | Description                                                                                   |
| ------- | --------------- | --------------------------------------------------------------------------------------------- |
| FE4ADD  | `a, b, c, 4, 4` | Set `[a:4]_4 = [b:4]_4 + [c:4]_4` with vector addition.                                       |
| FE4SUB  | `a, b, c, 4, 4` | Set `[a:4]_4 = [b:4]_4 - [c:4]_4` with vector subtraction.                                    |
| BBE4MUL | `a, b, c, 4, 4` | Set `[a:4]_4 = [b:4]_4 * [c:4]_4` with extension field multiplication.                        |
| BBE4DIV | `a, b, c, 4, 4` | Set `[a:4]_4 = [b:4]_4 / [c:4]_4` with extension field division. Division by zero is invalid. |

#### Hashes

The instructions below do block accesses with block size `1` and `CHUNK` in address space `4`.

| Name                                                                                                                                                                                                                  | Operands    | Description                                                                                                                                                                                                                                                                                                                                             |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| COMP_POS2 `[CHUNK, PID]` <br/><br/> Here `CHUNK` and `PID` are **constants** that determine different opcodes. `PID` is an internal identifier for particular Poseidon2 constants dependent on the field (see below). | `a,b,c,4,4` | Applies the Poseidon2 compression function to the inputs `[[b]_4:CHUNK]_4` and `[[c]_4:CHUNK]_4`, writing the result to `[[a]_4:CHUNK]_4`.                                                                                                                                                                                                              |
| PERM_POS2 `[WIDTH, PID]`                                                                                                                                                                                              | `a,b,_,4,4` | Applies the Poseidon2 permutation function to `[[b]_4:WIDTH]_4` and writes the result to `[[a]_4:WIDTH]_4`. <br/><br/> Each array of `WIDTH` elements is read/written in two batches of size `CHUNK`. This is nearly the same as `COMP_POS2` except that the whole input state is contiguous in memory, and the full output state is written to memory. |

The native extension uses the following Poseidon2 constants:

- `PID`: This identifier provides domain separation between different Poseidon2 constants. We use `0` to identify [
  `POSEIDON2_BABYBEAR_16_PARAMS`](https://github.com/HorizenLabs/poseidon2/blob/bb476b9ca38198cf5092487283c8b8c5d4317c4e/plain_implementations/src/poseidon2/poseidon2_instance_babybear.rs#L2023C20-L2023C48),
  but the Mat4 used is Plonky3's with a Montgomery reduction.
- `CHUNK`: We use `CHUNK = 8` for the native extension.
- `WIDTH`: We use `WIDTH = 16` for the native extension.

The input (of size `WIDTH`) is read in two batches of size `CHUNK`, and, similarly, the output is written in either one
or two batches of size `CHUNK`, depending on the output size of the corresponding opcode.

#### Proof Verification

We have the following special opcodes tailored to optimize FRI proof verification. They access address space `4`.

| Name                                                                                                                                                                                                                     | Operands        | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| VERIFY_BATCH `[CHUNK, PID]` <br/><br/> Here `CHUNK` and `PID` are **constants** that determine different opcodes. `PID` is an internal identifier for particular Poseidon2 constants dependent on the field (see below). | `a,b,c,d,e,f,g` | Further described [here](../../extensions/native/circuit/src/poseidon2/README.md). Due to already having a large number of operands, the address space is fixed to be `AS::Native = 4`. Computes `mmcs::verify_batch`. In the native address space, `[a], [b], [e], [f]` should be the array start pointers for the dimensions array, the opened values array (which contains more arrays) and the commitment (which is an array of length `CHUNK`). `[c]` should be the length of the opened values array (and so should be equal to the length of the dimensions array as well). `[d]` should be the hint id of proofs. `g` should be the reciprocal of the size (in field elements) of the values contained in the opened values array: if the opened values array contains field elements, `g` should be 1; if the opened values array contains extension field elements, `g` should be 1/4. |
| FRI_REDUCED_OPENING                                                                                                                                                                                                      | `a,b,c,d,e,f,g` | Let `a_ptr = [a]_4`, `b_ptr = [b]_4`, `length = [c]_4`, `alpha = [d:EXT_DEG]_4`, `hint_id = [f]_4`, `is_init = [g]_4`. `a_ptr` is the address of Felt array `a_arr` and `b_ptr` is the address of Ext array `b_arr`. Compute `sum((b_arr[i] - a_arr[i]) * alpha ^ i)` for `i=0..length` and write the value into `[e:EXT_DEG]_4`. It is required that `is_init` is boolean. If `is_init == 0`, read content of `a_arr` from the hint space at index `hint_id` and write into `a_arr`. Otherwise, read `a_arr` from memory.  This instruction removes elements from `hint_space[hint_id]` as they are read. |

#### Phantom Sub-Instructions

The native extension defines the following phantom sub-instructions.

| Name            | Discriminant | Operands      | Description                                                                                                                                                                                                                          |
| --------------- | ------------ | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| NativePrint     | 0x10         | `a,_,c_upper` | Prints `[a]_{c_upper}` to stdout on the host machine.                                                                                                                                                                                |
| NativeHintInput | 0x11         | `_`           | Pops a vector `hint` of field elements from `input_stream` and sets `hint_stream` to equal the vector `[[F::from_canonical_usize(hint.len())], hint].concat()`. The hint stream must previously be empty.                            |
| NativeHintBits  | 0x12         | `a,b,c_upper` | Sets `hint_stream` to be the least significant `b` bits of `([a]_{c_upper}).as_canonical_u32()`. The hint stream must previously be empty.                                                                                           |
| NativeHintLoad  | 0x13         | `_`           | Pops a vector `hint` of field elements from `input_stream` and appends it to `hint_space`. Sets `hint_stream` to contain a length-1 vector containing the index of `hint` in `hint_space`. The hint stream must previously be empty. |
| NativeHintFelt  | 0x14         | `_`           | Pops a field element from `input_stream` and set `hint_stream` equal to it. The hint stream must previously be empty.                                                                                                                |

### Keccak Extension

The Keccak extension supports the Keccak256 hash function. The extension operates on address spaces `1` and `2`, meaning
all memory cells are constrained to be bytes.

| Name           | Operands    | Description                                                                                                       |
| -------------- | ----------- | ----------------------------------------------------------------------------------------------------------------- |
| KECCAK256_RV32 | `a,b,c,1,2` | `[r32{0}(a):32]_2 = keccak256([r32{0}(b)..r32{0}(b)+r32{0}(c)]_2)`. Performs memory accesses with block size `4`. |

### SHA2-256 Extension

The SHA2-256 extension supports the SHA2-256 hash function. The extension operates on address spaces `1` and `2`,
meaning all memory cells are constrained to be bytes.

| Name        | Operands    | Description                                                                                                                                                              |
| ----------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| SHA256_RV32 | `a,b,c,1,2` | `[r32{0}(a):32]_2 = sha256([r32{0}(b)..r32{0}(b)+r32{0}(c)]_2)`. Does the necessary padding. Performs memory reads with block size `16` and writes with block size `32`. |

### BigInt Extension

The BigInt extension supports operations on 256-bit signed and unsigned integers. The extension operates on address
spaces `1` and `2`, meaning all memory cells are constrained to be bytes. Pointers to the representation of the elements
are read from address space `1` and the elements themselves are read/written from address space `2`. Each instruction
performs block accesses with block size `4` in address space `1` and block size `32` in address space `2`.

**Note:** These instructions are not the same as instructions on 256-bit registers.

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

### Algebra Extension

The algebra extension supports modular arithmetic over arbitrary fields and their complex field extensions. It is
configured to specify a list of supported moduli. The configuration of each supported positive integer modulus `N`
includes associated configuration parameters `N::NUM_LIMBS` and `N::BLOCK_SIZE` (defined below).

The instructions perform operations on unsigned big integers representing elements in the modulus. The extension
operates on address spaces `1` and `2`, meaning all memory cells are constrained to be bytes. Pointers to the
representation of the elements are read from address space `1` and the elements themselves are read/written from address
space `2`.

An element in the ring of integers modulo `N`is represented as an unsigned big integer with `N::NUM_LIMBS` limbs with
each limb having `LIMB_BITS = 8` bits. For each instruction, the input elements
`[r32{0}(b): N::NUM_LIMBS]_2, [r32{0}(c):N::NUM_LIMBS]_2` are assumed to be unsigned big integers in little-endian
format with each limb having `LIMB_BITS` bits. However, the big integers are **not** required to be less than `N`. Under
these conditions, the output element `[r32{0}(a): N::NUM_LIMBS]_2` written to memory will be an unsigned big integer of
the same format that is congruent modulo `N` to the respective operation applied to the two inputs.

For each instruction, the operand `d` is fixed to be `1` and `e` is fixed to be `2`.
Each instruction performs block accesses with block size `4` in address space `1` and block size `N::BLOCK_SIZE` in
address space `2`, where `N::NUM_LIMBS` is divisible by `N::BLOCK_SIZE`. Recall that `N::BLOCK_SIZE` must be a power of 2.

| Name                      | Operands    | Description                                                                                                                                                                                                |
| ------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ADDMOD_RV32\<N\>          | `a,b,c,1,2` | `[r32{0}(a): N::NUM_LIMBS]_2 = [r32{0}(b): N::NUM_LIMBS]_2 + [r32{0}(c): N::NUM_LIMBS]_2 (mod N)`                                                                                                          |
| SUBMOD_RV32\<N\>          | `a,b,c,1,2` | `[r32{0}(a): N::NUM_LIMBS]_2 = [r32{0}(b): N::NUM_LIMBS]_2 - [r32{0}(c): N::NUM_LIMBS]_2 (mod N)`                                                                                                          |
| SETUP_ADDSUBMOD_RV32\<N\> | `a,b,c,1,2` | `assert([r32{0}(b): N::NUM_LIMBS]_2 == N)` for the chip that handles add and sub. For the sake of implementation convenience it also writes something (can be anything) into `[r32{0}(a): N::NUM_LIMBS]_2` |
| MULMOD_RV32\<N\>          | `a,b,c,1,2` | `[r32{0}(a): N::NUM_LIMBS]_2 = [r32{0}(b): N::NUM_LIMBS]_2 * [r32{0}(c): N::NUM_LIMBS]_2 (mod N)`                                                                                                          |
| DIVMOD_RV32\<N\>          | `a,b,c,1,2` | `[r32{0}(a): N::NUM_LIMBS]_2 = [r32{0}(b): N::NUM_LIMBS]_2 / [r32{0}(c): N::NUM_LIMBS]_2 (mod N)`. Undefined behavior if `gcd([r32{0}(c): N::NUM_LIMBS]_2, N) != 1`.                                       |
| SETUP_MULDIVMOD_RV32\<N\> | `a,b,c,1,2` | `assert([r32{0}(b): N::NUM_LIMBS]_2 == N)` for the chip that handles mul and div. For the sake of implementation convenience it also writes something (can be anything) into `[r32{0}(a): N::NUM_LIMBS]_2` |

#### Modular Branching

The configuration of `N` is the same as above. For each instruction, the input elements
`[r32{0}(b): N::NUM_LIMBS]_2, [r32{0}(c): N::NUM_LIMBS]_2` are assumed to be unsigned big integers in little-endian
format with each limb having `LIMB_BITS` bits.

| Name                    | Operands    | Description                                                                                                                                                                                                                                                                                         |
| ----------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ISEQMOD_RV32\<N\>       | `a,b,c,1,2` | `[a:4]_1 = [r32{0}(b): N::NUM_LIMBS]_2 == [r32{0}(c): N::NUM_LIMBS]_2 (mod N) ? 1 : 0`. Enforces that `[r32{0}(b): N::NUM_LIMBS]_2, [r32{0}(c): N::NUM_LIMBS]_2` are less than `N` and then sets the register value of `[a:4]_1` to `1` or `0` depending on whether the two big integers are equal. |
| SETUP_ISEQMOD_RV32\<N\> | `a,b,c,1,2` | `assert([r32{0}(b): N::NUM_LIMBS]_2 == N)` in the chip that handles modular equality. For the sake of implementation convenience it also writes something (can be anything) into register value of `[a:4]_1`                                                                                        |

#### Phantom Sub-Instructions


| Name           | Discriminant | Operands      | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| -------------- | ------------ | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| HintNonQr\<N\>  | 0x50         | `_,_,c_upper` | Use `c_upper` to determine the index of the modulus from the list of supported moduli. Reset the hint stream to equal a quadratic nonresidue modulo `N`. |
| HintSqrt\<N\>   | 0x51         | `a,_,c_upper` | Use `c_upper` to determine the index of the modulus from the list of supported moduli. Read from memory `x = [r32{0}(a): N::NUM_LIMBS]_2`.  If `x` is a quadratic residue modulo `N`, reset the hint stream to `[1u8, 0u8, 0u8, 0u8]` followed by a square root of `x`.  If `x` is not a quadratic residue, reset the hint stream to `[0u8; 4]` followed by a square root of `x * non_qr`, where `non_qr` is the quadratic nonresidue returned by `HintNonQr<N>`. |

#

#### Complex Extension Field

A complex extension field `Fp2` is the quadratic extension of a prime field `Fp` with irreducible polynomial `X^2 + 1`.
An element in `Fp2` is a pair `c0: Fp, c1: Fp` such that `c0 + c1 u` represents a point in `Fp2` where `u^2 = -1`.

The complex extension field `Fp2` is supported only if the modular arithmetic instructions for `Fp::MODULUS` is also
supported.
The memory layout of `Fp2` is then that of two concatenated `Fp` elements,
and the block size for memory accesses is the block size of `Fp`.

We use the following notation below:

```
r32_fp2(a) -> Fp2 {
    let c0 = [r32{0}(a): Fp::NUM_LIMBS]_2;
    let c1 = [r32{0}(a) + Fp::NUM_LIMBS: Fp::NUM_LIMBS]_2;
    return Fp2 { c0, c1 };
}
```

| Name                | Operands    | Description                                                                                                                                                                  |
| ------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ADD\<Fp2\>          | `a,b,c,1,2` | Set `r32_fp2(a) = r32_fp2(b) + r32_fp2(c)`                                                                                                                                   |
| SUB\<Fp2\>          | `a,b,c,1,2` | Set `r32_fp2(a) = r32_fp2(b) - r32_fp2(c)`                                                                                                                                   |
| SETUP_ADDSUB\<Fp2\> | `a,b,c,1,2` | `assert([r32_fp2(b).c0 == N)` for the chip that handles add and sub. For the sake of implementation convenience it also writes something (can be anything) into `r32_fp2(a)` |
| MUL\<Fp2\>          | `a,b,c,1,2` | Set `r32_fp2(a) = r32_fp2(b) * r32_fp2(c)`                                                                                                                                   |
| DIV\<Fp2\>          | `a,b,c,1,2` | Set `r32_fp2(a) = r32_fp2(b) / r32_fp2(c)`                                                                                                                                   |
| SETUP_MULDIV\<Fp2\> | `a,b,c,1,2` | `assert([r32_fp2(b).c0 == N)` for the chip that handles mul and div. For the sake of implementation convenience it also writes something (can be anything) into `r32_fp2(a)` |

### Elliptic Curve Extension

The elliptic curve extension supports arithmetic over elliptic curves `C` in Weierstrass form given by
equation `C: y^2 = x^3 + C::A * x + C::B` where `C::A` and `C::B` are constants in the coordinate field. We note that
the definitions of the
curve arithmetic operations do not depend on `C::B`. The VM configuration will specify a list of supported curves. For
each Weierstrass curve `C` there will be associated configuration parameters `C::COORD_SIZE` and `C::BLOCK_SIZE` (
defined below). The extension operates on address spaces `1` and `2`, meaning all memory cells are constrained to be
bytes.

An affine curve point `EcPoint(x, y)` is a pair of `x,y` where each element is an array of `C::COORD_SIZE` elements each
with `LIMB_BITS = 8` bits. When the coordinate field `C::Fp` of `C` is prime, the format of `x,y` is guaranteed to be
the same as the format used in the [modular arithmetic instructions](#modular-arithmetic). A curve point will be
represented as `2 * C::COORD_SIZE` contiguous cells in memory.

We use the following notation below:

```
r32_ec_point(a) -> EcPoint {
    let x = [r32{0}(a): C::COORD_SIZE]_2;
    let y = [r32{0}(a) + C::COORD_SIZE: C::COORD_SIZE]_2;
    return EcPoint(x, y);
}
```

| Name                 | Operands    | Description                                                                                                                                                                                                                                                                                    |
| -------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| EC_ADD_NE\<C\>       | `a,b,c,1,2` | Set `r32_ec_point(a) = r32_ec_point(b) + r32_ec_point(c)` (curve addition). Assumes that `r32_ec_point(b), r32_ec_point(c)` both lie on the curve and are not the identity point. Further assumes that `r32_ec_point(b).x, r32_ec_point(c).x` are not equal in the coordinate field.           |
| SETUP_EC_ADD_NE\<C\> | `a,b,c,1,2` | `assert(r32_ec_point(b).x == C::MODULUS)` in the chip for EC ADD. For the sake of implementation convenience it also writes something (can be anything) into `[r32{0}(a): 2*C::COORD_SIZE]_2`. It is required for proper functionality that `assert(r32_ec_point(b).x != r32_ec_point(c).x)`   |
| EC_DOUBLE\<C\>       | `a,b,_,1,2` | Set `r32_ec_point(a) = 2 * r32_ec_point(b)`. This doubles the input point. Assumes that `r32_ec_point(b)` lies on the curve and is not the identity point.                                                                                                                                     |
| SETUP_EC_DOUBLE\<C\> | `a,b,_,1,2` | `assert(r32_ec_point(b).x == C::MODULUS)` in the chip for EC DOUBLE. For the sake of implementation convenience it also writes something (can be anything) into `[r32{0}(a): 2*C::COORD_SIZE]_2`. It is required for proper functionality that `assert(r32_ec_point(b).y != 0 mod C::MODULUS)` |

### Pairing Extension

The pairing extension supports opcodes tailored to accelerate pairing checks using the optimal Ate pairing over certain
classes of pairing friendly elliptic curves. For a curve `C` to be supported, the VM must have enabled instructions for
`C::Fp` and `C::Fp2`. The memory block size is `C::Fp::BLOCK_SIZE` for both reads and writes. The currently supported
curves are BN254 and BLS12-381. The extension operates on address spaces `1` and `2`, meaning all memory cells are
constrained to be bytes.

#### Phantom Sub-Instructions

The pairing extension defines the following phantom sub-instructions.

| Name         | Discriminant | Operands      | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ------------ | ------------ | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| HintFinalExp | 0x30         | `a,b,c_upper` | Uses `c_upper = PAIRING_IDX` to determine the curve: `BN254 = 0, BLS12-381 = 1`. `a` is a pointer to `(p_ptr, p_len): (u32, u32)` in memory, and `b` is a pointer to `(q_ptr, q_len): (u32, u32)` in memory (e.g., `p_ptr = [r32{0}(a)..r32{0}(a) + 4]_2`). The sub-instruction peeks at `P = [p_ptr..p_ptr + p_len * size_of<Fp>() * 2]_2` and `Q = [q_ptr..q_ptr + q_len * size_of<Fp2>() * 2]_2` and views `P` as a list of `G1Affine` elements and `Q` as a list of `G2Affine` elements. It computes the multi-Miller loop on `(P, Q)` and then the final exponentiation hint `(residue_witness, scaling_factor): (Fp12, Fp12)`. It resets the hint stream to equal `(residue_witness, scaling_factor)` as `NUM_LIMBS * 12 * 2` bytes. |

## Acknowledgements

The design of the native extension was inspired by [Valida](https://github.com/valida-xyz/valida-compiler/issues/2) with
changes suggested by Max Gillet for compatibility with existing ISAs.
