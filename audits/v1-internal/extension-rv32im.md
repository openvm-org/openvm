# VM Extension: RV32IM and IO

Author: [manh9203](https://github.com/manh9203)

## 1. Introduction

### 1.1 Scope

[RV32IM and IO extension](https://github.com/openvm-org/openvm/blob/main/extensions/rv32im/)

### 1.2 Commit

<https://github.com/openvm-org/openvm/commit/f038f61d21db3aecd3029e1a23ba1ba0bb314800>

### 1.3 Describe the main focus and any additional context

- Verify that the RV32IM-related instructions in the OpenVM ISA conform to the framework

- Examine the RV32IM and IO transpilers to OpenVM
  - Prove that, assuming the invariants of the OpenVM ISA hold, the RV32IM specification is fully upheld. This is achieved by thoroughly reviewing each instruction.
  - Address aspects of IO and serialization/deserialization (serde)

- Reiterate the handling of the `x0` register

## 2. Findings

Findings include anything that could warrant change or unexpected behavior that should be brought to light. They range from severe to informational.

### 2.1 Missing constraints on address spaces for `LOAD/STORE` instructions

**Severity:** High

**Context:** Checking if the implementation of `LOAD/STORE` instructions upholds OpenVM ISA invariants

**Description:** At the circuit level, we need to assert supported address spaces of `LOAD/STORE` instructions to avoid breaking invariants of each address space

**Recommendation:** Add constraints on address spaces for LOAD/STORE instructions:

- `mem_as` in `LOAD` opcodes in `RV32IM` extension should be in `{0, 1, 2}`
- `mem_as` in `STORE` opcodes in `RV32IM` extension should be in `{2, 3, 4}`

**Resolution:** <https://github.com/openvm-org/openvm/commit/171ec20ffea9fed8c67292d4d91dfb9236029ef1>

Added constraints for `mem_as` in `Rv32LoadStoreAdapterChip` to avoid
breaking invariants of each address space.
- `mem_as` for `LOAD` instructions should be in `0/1/2`
- `mem_as` for `STORE` instructions should be in `2/3/4`

Added constraints for `is_load` and `is_valid` in
`LoadSignExtendCoreChip` and `LoadStoreCoreChip`.

### 2.2 Incorrect constraint for `write_data` in `STORE` instruction

**Severity:** Medium

**Context:** Checking circuit implementation of `STORE` instruction

**Description:** There seems to be a typo [here](https://github.com/openvm-org/openvm/blob/f038f61d21db3aecd3029e1a23ba1ba0bb314800/extensions/rv32im/circuit/src/loadstore/core.rs#L197). This did not correctly constrain the last 2 cells of `read_data`'s half word.

**Recommendation:** The fix depends on how we define half word when `NUM_CELLS > 4`.

- If we define it as 2 bytes, then we should change `i + 2` to `i - 2`
- If we define it as `NUM_CELLS / 2` bytes, then we should change the if condition to:

```rust
if i >= NUM_CELLS / 2 {
    read_data[i - NUM_CELLS / 2]
} else {
    prev_data[i]
}
```

**Resolution:** <https://github.com/openvm-org/openvm/pull/1406>
https://github.com/openvm-org/openvm/commit/df041e70e55999626d7515cce883f894a160d4b4

### 2.3 Constraint on `divrem` chip does not match code comments

**Severity:** Informational

**Context:** Checking circuit implementation of `DIVREM` instruction

**Description:** [This comment](https://github.com/openvm-org/openvm/blob/f038f61d21db3aecd3029e1a23ba1ba0bb314800/extensions/rv32im/circuit/src/divrem/core.rs#L217-L218) says that we constrain `q` to be non-zero if `q_sign == 1`. However, [this constraint](https://github.com/openvm-org/openvm/blob/f038f61d21db3aecd3029e1a23ba1ba0bb314800/extensions/rv32im/circuit/src/divrem/core.rs#L237-L245) does not enforce that. It only constrains `q` is zero and `q_sign == 0` if `q_sign != sign_xor`. So there could be a case that `q = 0` and `q_sign == sign_xor == 1` that can still pass the constraint.

**Recommendation:** To satisfy the constraint `b = c * q + r`, `q_sign` cannot be `1` if `q = 0`. So we should just change the code comment to explain that. And we might remove the second constraint as it is implied by the first one.

**Resolution:** <https://github.com/openvm-org/openvm/pull/1406>
https://github.com/openvm-org/openvm/commit/df041e70e55999626d7515cce883f894a160d4b4

### 2.4 Code comments typos in `base_alu` and `io.libs`

**Severity:** Informational

**Context:** Checking implementation of RV32IM extension

**Description:** There are some typos in code comments of `base_alu` and `io.libs`

**Recommendation:** Fix the typos:

- [`base_alu`](https://github.com/openvm-org/openvm/blob/f038f61d21db3aecd3029e1a23ba1ba0bb314800/extensions/rv32im/circuit/src/base_alu/core.rs#L96): should be `a[i] + c[i] - b[i]`
- [`io.libs`](https://github.com/openvm-org/openvm/blob/f038f61d21db3aecd3029e1a23ba1ba0bb314800/extensions/rv32im/guest/src/io.rs#L58): should be `[[rd] + imm]_3`

**Resolution:** <https://github.com/openvm-org/openvm/pull/1404>
https://github.com/openvm-org/openvm/commit/b0ef81782b0b78268627308fdfb455822259099b

## 3. Discussion

### 3.1 Constraints on Instructions's parameters

In the current circuit implementation, variables retrieved directly from the instructions don't require additional constraints because they are already constrained by the program bus.

For example, in `loadstore` adapter, we have:

```rust
self.execution_bridge
    .execute(
        ctx.instruction.opcode,
        [
            local_cols.rd_rs2_ptr.into(),
            local_cols.rs1_ptr.into(),
            local_cols.imm.into(),
            AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
            local_cols.mem_as.into(),
            local_cols.needs_write.into(),
            local_cols.imm_sign.into(),
        ],
        local_cols.from_state,
        ExecutionState {
            pc: to_pc,
            timestamp: timestamp + AB::F::from_canonical_usize(timestamp_delta),
        },
    )
    .eval(builder, is_valid);
```

Here, `imm` and `imm_sign` are not constrained in the adapter’s `eval` function because they must match the actual parameters of the executing instruction. Ensuring that these parameters adhere to the specs is the transpiler’s responsibility.

However, we still need to carefully constrain any variable derived from the instruction parameters. For example, if we decompose `imm` into `imm_limbs` and use it to constrain the sum of two numbers, we must still perform a range check on each limb of `imm_limbs`, because these values come from the prover and cannot be fully trusted.

### 3.2 Constraints on `ctx` variables in adapters

In adapter implementations, variables from `AdapterAirContext` are not constrained within the `eval` function. We assume that these variables—passed from the core air—are already constrained there. It's important to keep this in mind whenever we write a new `VmWrapperChip`.

### 3.3 `x0` handling

The RISC-V transpiler already converts any instruction that writes to `x0` into a `nop`. However, there are exceptions for `load/store` and `jal/jalr` instructions, since those writes are handled in the adapters.

For these instructions, we use the instruction parameter `f` to check if the write register is `x0`. If it is, we skip the write in the adapter’s postprocess function. Currently, we only use the `f` parameter for this purpose. If we repurpose it in the future, we should verify that it doesn’t interfere with the `x0` handling of these instructions.
