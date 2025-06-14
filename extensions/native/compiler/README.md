# Native Compiler

Native compiler is mainly designed for a human friendly DSL(domain-specific language) which can write a stark
verification program.

`Builder<C: Config>` is the main struct of the native compiler. Users should use this struct for their DSL.

# Compiler Mode

Native compiler supports two modes:

- Static mode: the program is compiled into a Halo2 circuit. This mode doesn't support jump and heap allocation. This
  means that the program always has a **fixed-length** execution trace. In this mode, all loops are unrolled at compile
  time.
- Dynamic mode: the program is compiled into an OpenVM executable. This mode supports jump and heap allocation.
  This means that the program could have a **variable-length** execution trace.

# DSL

A DSL program is an intermediate representation of the target program, which could be a OpenVM `Program` using Native
extension in dynamic mode or a Halo2 circuit in static mode. The motivation is:

- DSL is more human friendly.
- Unify the programs in static and dynamic mode.

Users can use `Builder` to build their DSL program, which is a list of DSL instructions. All DSL instructions could be
found in `src/ir/instructions.rs`. Users use `AsmCompiler` in `src/asm/compiler.rs` or
`Halo2Compiler` in `src/halo2/compiler.rs` to compile the DSL program into the target program.

Most DSL instructions are supported in both static and dynamic mode, except:

- Instructions with `Circuit` prefix are only supported in static mode.
- Memory instructions(e.g. `Load*`/`Store*`) are only supported in dynamic mode.
- Control flow instructions are only supported in dynamic mode because **they must be unrolled in static mode**.
- Hint instructions are only supported in dynamic mode.
- Witness instructions are only supported in static mode.
- `Poseidon2PermuteBabyBear`/`Poseidon2CompressBabyBear`/`FriReducedOpening`/`VerifyBatchFelt`/`VerifyBatchExt` are only
  supported in dynamic mode.

Usually users don't need to know details about DSL instructions and only need to call functions of `Builder`. Sometimes
users may want to give different implementation in static/dynamic mode. In this case, they could check
`builder.flags.static_only`.

For now the DSL syntax doesn't support defining functions.

## Control Flow
`Builder` unrolls loops/branches if the condition can be evaluated at compile time.

Be careful when using assignments in loops/branches. Use `set_value`/`assign` which can be executed at runtime instead
of Rust assignment(`=`), which can only be executed at compile time. The latter usually has undesired effects.


# Variable

`Variable` trait in `src/ir/var.rs` defines the concept of a variable in the DSL. If a type is a `Variable`:

- It supports `uninit` which declares an uninitialized variable, which doesn't have any runtime cost until it's used.
- It supports `assert_eq` which asserts that two variables are equal.
- It associates with a `Expression` type, which is the expression of this type. This is mostly for arithmetic operations
of basic data types because symbolic expressions need another type.
- A variable can be assigned with value of an expression.

## MemVariable

**Dynamic mode only**

The dynamic mode uses memory model with stack and heap. That requires that a variable can be moved between stack and heap. 
Therefore, we introduce `MemVariable` trait in `src/ir/var.rs`, which extends `Variable` and defines the interfaces to 
load/store a variable.

# Data Types

## Basic Data Types

- `Var<C::N>`: the natural field of the stark to prove the execution of this program.
- `Felt<C::F>`: the base field of the stark to be verified.
- `Ext<C::F, C::EF>`: the extended field of the stark to be verified.

The basic types are bundled as a trait, `Config` in `src/ir/mod.rs`. `Config` is a generic parameter for `Builder`.

In compiler implementation, variables of each basic data type is assigned an ID.

## CR Variable

To unify static/dynamic programs in eDSL, we introduce a new concept, CR variable("C" for compile time and "R" for
runtime). A CR variable behaves like a normal variable in eDSL(e.g. it supports assignment), but it could be either a
constant or a variable in runtime.

Currently, there are 2 types of CR variables:

### Usize

`Usize` represents:

- A `Var` in the dynamic mode(variant `Var`).
- A compile-time variable in the static mode(variant `Const`).

`Usize` is usually used as loop variables and array length.

ATTENTION: In static mode, multiple `Usize`s could point to the same variable. Usually this happens when cloning
`Usize`. For example, `a = b.clone()` and `b` is modified. Users may not expect update of `a` because no modification is
made to that symbol.

### Array

`Array` represents:

- A variable-length array in the dynamic mode(variant `Dyn`). It is allocated on the heap. `Dyn` supports dynamic index access.
- A fixed-length array of **variable references** in the static mode(variant `Fixed`). It doesn't allocate anything. 
`Fixed` only supports constant index access.
  

ATTENTION: In static mode, multiple elements of an `Array` could point to the same variable. Be careful when calling
`builder.set_value` which doesn't create a new variable. 
For example:
```
let a = builder.eval(0);
let b = builder.array(2);
builder.set_value(&b, 0, a);
builder.set_value(&b, 1, a);
```
`b[0]` and `b[1]` will point to the same variable. The value of `b[1]` will also change when `b[0]` is assigned.

## Ptr

**Dynamic mode only**

A wrapper type of `Var` to represent a pointer.

## DslVariable Macro

`DslVariable` is a macro to derive `Variable` and `MemVariable` for a struct composed of types implementing `Variable`
and `MemVariable`.

# Expression

`SymbolicVar`/`SymbolicFelt`/`SymbolicExt` are the `Expression`s of basic types `Var`/`Felt`/`Ext`. These types only 
exist at compile time and only could create instructions when they are evaluated.

`Expression` of most other types is themselves because they don't need arithmetization. 

## Evaluation

Currently `builder.eval` always creates a new variable. When the expression is a constant or a single variable, using
`builder.eval` to evaluate an expression causes an unnecessary variable assignment, which impacts performance
significantly.

The naming of `builder.eval` is semantically incorrect because creating a new variable is out of scope of "eval".
However, the assumption is taken too widely to fix. So we introduce a new method `builder.eval_expr` for the real
"evaluation".

`builder.eval_expr` returns a **[right value](https://www.oreilly.com/library/view/c-in-a/059600298X/ch03s01.html#:~:text=The%20term%20rvalue%20is%20a,are%20close%20to%20the%20truth.)**, which is either a constant or a variable. `builder.eval_expr` 
avoids creating a new variable if possible. An instruction can use a right value for reading a value directly.

Currently `builder.eval_expr` is not generic. Its input must be `SymbolicVar`. `RVar` is the type of the return value of
`builder.eval_expr`.

# AsmCompiler

ASM compiler is almost a normal compiler except it doesn't use registers(usually address space 3 in RISC-V) and only 
uses a dedicated address space(4) as memory. It compiles a DSL program into a OpenVM `Program` with a list of native extension instructions.

## Memory Layout

Asm Compiler uses a common memory layout:

|          |
|----------|
| STACK    |
| A0       |
| HEAP_PTR |
| STACK    |

`HEAP_PTR`/`A0` are 2 reserved addresses for memory allocation operations. Because heap grows downwards and stack grows 
upwards, regular operations in DSL cannot touch `HEAP_PTR`/`A0` unless users intend to do so.

Start addresses of each segment are constants in `src/asm/compiler.rs`.

## Stack Variable
Frame pointer of each basic data type variable can be computed based on their IDs: 
- `Var`s are stored in stack positions 1+8*0, 2+8*0, 1+8*1, 2+8*1, 1+8*2, 2+8*2, ...
- `Felt`s are stored in stack positions 3+8*0, 4+8*0, 3+8*1, 4+8*1, 3+8*2, 4+8*2, ...
- `Ext`s are stored in stack positions 5-8, 13-16, 21-24, ...

Because DSL doesn't support functions, all pointers of variables are known at compile time. ASM compiler can avoid stack
overflow at compile time.

## Memory Allocation
Memory allocation is done by moving the value at `HEAP_PTR` forward with the help of `A0`. Currently, there is no 
de-allocation at compiler-level. But users can de-allocate by manually resetting the value at `HEAP_PTR`.

**[!WARNING]**

The top of the memory is `2^29`. Trying to allocate more than available memory will
cause a runtime error and cannot generate a valid proof.

## Control Flow
Asm Compiler supports both loops and branches like normal compilers.

# Halo2Compiler

Halo2 compiler compiles a DSL program into a Halo2 circuit, which doesn't support heap and jump opcodes.

## Stack Variable
Each variable is tracked by an assigned ID. Halo2 compiler keeps a mapping from ID to `AssignedValue`.

## Control Flow
Users can still use loops and branches in `Builder`. But loops and branches must be unrolled at 
compile time.

# Debug Backtraces
When a native program terminates unexpectedly, backtraces are helpful for debugging. 

Requirements to print out backtraces correctly:
1. The binary which compiles the native program must be the binary to execute the native program. This is because the 
backtraces depend on the symbol table of the binary.
2. The binary must be compiled with `debug=2`, otherwise the symbol table is not complete.
3. When the native program is compiled, `RUST_BACKTRACE=1` must be set.
4. When the native program is executed, `profiling` flag must be set in `VmConfig`.