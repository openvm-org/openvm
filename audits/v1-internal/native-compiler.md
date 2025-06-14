# Native Compiler

Author: [Xinding Wei](https://github.com/nyunyunyunyu)

## 1. Introduction

Scope: https://github.com/openvm-org/openvm
Commit: 336f1a475e5aa3513c4c5a266399f4128c119bba

## 2. Findings

### 2.1 `alloc` doesn't check overflow

**Severity:** Medium
**Context:** https://github.com/openvm-org/openvm/blob/336f1a475e5aa3513c4c5a266399f4128c119bba/extensions/native/compiler/src/asm/compiler.rs#L598

**Description:** When allocating memory, `HEAP_PTR` and `A0` could overflow as a field element. This could lead
an exploit when the size of memory allocation is based on inputs.

The exploit could change `HEAP_PTR` to an arbitrary address, which could point to a loop variable or an
end condition. The the exploit could write an arbitrary value into the address and takes control of the
control flow.

**Proof of concept:** N/A

**Recommendation:** Add a range check instruction in order to avoid overflow.

**Resolution:** https://github.com/openvm-org/openvm/commit/bb07891747bae8aace4a0e8ea0b2089548a06ff8

- Add a new opcode `RANGE_CHECK`. `RANGE_CHECK a, b, c` asserts that
`[a]_4 = x + y * 2^16` for some `x < 2^b` and `y < 2^c`. In order to
save columns, `RANGE_CHECK` is put into the existing `JalChip`.
- Update `alloc` in ASM compiler to enforce `HEAP_PTR` not overflow.
- Add `builder.range_check_v` to range check in DSL.

### 2.2 Assertion in dynamic mode doesn't result in real constraints in circuits

**Severity:** High
**Context:** https://github.com/openvm-org/openvm/blob/336f1a475e5aa3513c4c5a266399f4128c119bba/extensions/native/compiler/src/conversion/mod.rs#L274

**Description:**
ASM compiler compiles `Assert*` DSL instructions into a conditional jump + a ASM instruction `Trap`, which only results a phantom instruction. The exploit can generate a valid execution trace which ignores all assertions in the program.

**Proof of concept:** N/A

**Recommendation:** Add a `Terminate` instruction with exit code = 1 after `Trap`.

**Resolution:** https://github.com/openvm-org/openvm/commit/379476f8a9730d2a37d49ffbdc03d85c3416e68f

Terminate the program with exit code 1. The proof cannot bypass
assertions anymore.

### 2.3 Bits representation in CircuitNum2BitsV could overflow Bn254Fr
**Severity:** Medium
**Context:**: https://github.com/openvm-org/openvm/blob/336f1a475e5aa3513c4c5a266399f4128c119bba/extensions/native/compiler/src/constraints/halo2/compiler.rs#L317

**Description:**
The order of `Bn254Fr` is less than `2^254`. A number of 254 bits could overflow. Therefore the bit decomposition
of a specific `Bn254Fr` doesn't guarantee an unique representation.

**Recommendation:**
Constraints the bit representation is not in `[p, 2^254)` where `p` is the order of `Bn254Fr`.

**Resolution:** https://github.com/openvm-org/openvm/commit/bff6d573ce7e5304fed5a9e40df9a76647be42ea

### 2.4 ASM compiler could compile stackoverflow programs
**Severity:** Low
**Context:**: https://github.com/openvm-org/openvm/blob/336f1a475e5aa3513c4c5a266399f4128c119bba/extensions/native/compiler/src/asm/compiler.rs#L40

**Description:**
In compiled programs, frame pointers could be negative, which means stackoverflow. Usually compilers support
recursion so they cannot check stackoverflow at compile time. But ASM compiler can determine all frame pointers
at compile time so it has the ability to check.

This exploit can happen only when users create lots of stack variables and never access stack variables in
out of bound addresses(>=`2^29`). So it's very unlikely unless users are malicious.

**Recommendation:**
Assert frame pointers cannot be negative.

**Resolution:** https://github.com/openvm-org/openvm/pull/1416
https://github.com/openvm-org/openvm/commit/b02c1bdff3d97c69ffa9fa8d39769fdbf05a91de

All frame pointers are known at compile time. So we can check stack overflow at compile time.

## 3. Discussion

### 3.1 Analysis of ASM Compiler
The ASM compiler in `src/asm` converts DSL instructions into ASM instructions. The ASM compiler assumes `N` and `F` in config are the same type.

Most DSL instructions are trivially converted into the corresponding ASM instructions. Here we highlight some non-trivial instructions:
- `If*`. This kind of instructions append 1 branch instruction before the then/else closure.
- `ZipFor` appends 1 branch instruction after the loop body.
- `Alloc` computes the allocation size then increases `HEAP_PTR` by the allocation size. Finding `2.1` about this.
- `Assert*` results an if branch which panics if the condition is satisfied. Finding `2.2` about this.
- Debug instructions like `Print*`/`CycleTracker*`/

Notably, immediate `Ext` DSL instructions result 5 ASM instruction - the compiler needs to write the immediate `Ext` as 4 `Felt`s first.

### 3.2 Analysis of Halo2 Compiler
The Halo2 compiler in `src/asm` converts DSL instructions into Halo2 circuit constraints. The Halo2 compiler
doesn't support jump and heap allocation. So it's simpler than the ASM compiler. Almost all DSL instructions
are simply converted into the corresponding Halo2 circuit constraints.

Non-trivial instructions:
- `BabyBearChip` operations. https://github.com/openvm-org/openvm/pull/1407 add more explanation how it works.
- Poseidon2 operations. The implementation is copied from an audited codebase.
- Bit decomposition. Had a finding in 2.3.
