# OpenVM frontend and RVR preflight context

This file collects the context supplied for the RVR preflight design so another
reviewer can evaluate `rvr-preflight.md` without needing the original conversation.

## Original proposal: separating execution and tracegen

Currently, OpenVM execution is structured such that pure execution is independent
of tracegen (and chip design), but metered and especially preflight execution depend
heavily on the chip design.

The current model is:

```text
Opcode -> Chip -> Execute
```

The proposed inversion is:

```text
Opcode -> Execute -> Chip
```

The current structure distorts our thinking because execution optimizations cannot
be pursued cleanly at the opcode level; execution is always viewed through chips.

### Pure execution

For best performance, execution should be AOT- or JIT-compiled because interpreter
overhead is significant. The current x86 AOT path is performant but:

- is not portable to ARM;
- is difficult to profile beyond aggregate IPS/time;
- requires handwritten x86 assembly expertise;
- cannot benefit naturally from LLVM/GCC improvements.

Emitting C should achieve near-assembly performance while being portable and
maintainable. More importantly, execution should be organized around compiler basic
blocks rather than individual instructions. Reference:
[basic block](https://en.wikipedia.org/wiki/Basic_block).

```text
Traditional: instruction -> instruction -> branch -> instruction

Block based: Block A (instructions ending in a branch) -> Block B or Block C
```

Dynamic `JALR` targets complicate block discovery because arbitrary targets imply
a possible block at every address. Most compiler-generated jumps are not truly
arbitrary, and edge cases can be handled specially. PolkaVM disallows absolute
dynamic jumps; this is related to jump-destination analysis but is a separate
problem.

Meaningful execution metrics are:

- number of RISC-V instructions executed;
- number of native instructions per RISC-V instruction.

Wall-clock time and IPS are useful for relative tracking but combine guest
complexity, compiler efficiency, translation quality, and hardware. Precompiles
complicate native-per-guest-instruction accounting and need separate treatment.

The supplied target output looked like:

```text
exit code: 0
final pc:  0x801434F2
insns:     7.53B guest, 13.28B host (1.76x)
time:      749.12ms
speed:     10.05BIPS
```

### Metered execution

Metered execution should either disappear or become a single-dimensional gas
metric that approximates memory use, rather than today's multidimensional
chip-shaped accounting.

### Current preflight problem

The current pipeline is:

```text
Opcode -> Chip -> Execute -> Per-chip record -> GPU -> Tracegen
```

Preflight creates a record for each opcode/chip pairing. The one-to-one opcode to
chip mapping is already problematic for designs such as Keccak. Making opcode to
record one-to-many would deepen the coupling; the preferred direction is to remove
records entirely.

### Proposed preflight model

Preflight should be normal execution plus append-only logging of state accesses:

```text
Program log: [(timestamp, pc), ...]
Memory log:  [(timestamp, address, value), ...]
```

The program log can be partitioned after execution by opcode. GPU tracegen receives
the shared memory log and re-executes instructions in parallel. Instead of reading
mutable state, each trace generator reads the values recorded for the instruction's
logical time.

Conceptually:

```c
fn kernel(instr: &[Instruction], pc: &[u32], timestamp: &[u32], log: &[MemoryLog]) {
    // Re-execute each instruction, reading memory from the immutable log.
}
```

This is similar in spirit to
[`ReadOnlyTranscript`](https://github.com/openvm-org/stark-backend/blob/c92142ef671cfa1d556c4c291960c8c70f4aa3be/crates/stark-backend-v2/src/poseidon2/sponge.rs#L291-L296)
in stark-backend-v2. The memory log is sent once to the GPU and shared by all
tracegen kernels.

The simplified pipeline is:

```text
Opcode -> Execute -> Logs -> GPU -> Per-chip tracegen
```

Continuations and memory-Merkle construction add complexity but are orthogonal to
execution and should be handled separately between execution and tracegen.

With this split:

- pure execution is primarily host-compute-bound;
- preflight should approach an append/store-bound host workload.

That performance hypothesis must be measured; logging writes can overlap with
other CPU instructions and GPU access patterns may be non-coalesced.

## Precompile design context

The desired direction is bottom-up and profile-driven:

```text
"This block is hot" -> "Can these instructions become a super-instruction?"
```

rather than:

```text
"Crypto is expensive" -> patch a library -> special-case it in the VM
```

The Keccak-f loop is an example of a large, hot basic block that is a natural
super-instruction candidate. The concrete example supplied was the
`0x8020b17a-0x8020b952` loop: 674 instructions, with the back-edge remaining inside
the same generated C function. The broader goal is automatic or semi-automatic
precompile discovery:

```text
Profile hotspots -> identify patterns -> generate super-instruction -> integrate
       ^                                                        |
       +----------------------- feedback -----------------------+
```

The secp256k1 affine-operation slowdown is viewed as self-inflicted by the current
precompile/guest-library design; the original projective guest code was not
necessarily bottlenecked by inversion. Pairing hints should reduce circuit cost
without slowing execution; routing hints through memory into chip records is a
design flaw.

## Other frontend concerns

- The repository is brittle and difficult to experiment in; small changes can
  break unrelated areas and compile times are high.
- The execution stack is over-abstracted: by the time an opcode executes, its
  origin and semantics are difficult to profile.

## Requested RVR preflight design

The design should be grounded in the current RVR and interpreter-preflight code and
should ultimately remove `RecordArena` entirely.

The first implementation step is only the RVR preflight executor. It must record
the minimum data required to transform serial execution over read/write memory into
parallel trace re-execution over read-only data. A rough mental model is:

```text
mutable memory before: (address, value)
preflight memory log:  (timestamp, address, value)
program log:           (timestamp, pc)
```

Timestamp here means the OpenVM logical memory clock, not simply instruction
ordinal. Some operations, such as memory peeks, do not increment it. The design
must decide how such operations are replayed without turning them into fake timed
memory accesses.

The RVR implementation should reuse useful value-tracing/code-generation seams,
but not inherit chip-shaped records. After the executor is benchmarked against
interpreter preflight, GPU tracegen for the RISC-V AIRs is the first feasibility
test. GPU system/continuation tracegen follows, then other extensions, with CPU
tracegen only if it remains useful. Only after every production consumer has
migrated should `RecordArena` and interpreter preflight be deleted.

The design should be idiomatic, clean, minimal, and performance-first. It should be
reviewed adversarially against current RVR execution, memory semantics, every
tracegen consumer, system AIRs, continuations, migration risk, and prior RVR
preflight attempts.

## Follow-up corrections and constraints

### Immediate operands do not consume timestamp slots

In the current RV64 implementation, no active AIR consumes a separate timestamp
slot merely because an operand is immediate.

- Immediate ALU operations use an `rs1` read and `rd` write. The immediate travels
  on the execution bus with `RV64_IMM_AS`.
- Branch immediates are execution-bus operands; only the two register reads tick.
- Load/store offsets are algebraic operands; timestamps belong to registers and
  memory accesses.
- JALR's immediate does not tick; the `rs1` read and `rd` write/disabled-write gap
  do.
- LUI, AUIPC, and JAL consume only their destination-write slot.

`tracing_read_imm()` incremented the timestamp but had no call sites and was stale
scaffolding. It has been removed.

The live clock-without-memory-event cases are:

- disabled or `x0` destination writes;
- the unused second block of a non-crossing load/store;
- hint-store's fixed clock schedule;
- phantom instructions.

Peeks consume no timestamp.

### Minimize every preflight write

The amount written by preflight directly affects its speed. The design should not
add fields merely to make a durable or self-describing artifact. In particular:

- no schema version;
- no program digest;
- no VM-config digest;
- no transcript header object;
- no per-instruction access or observation cursors if they can be derived later;
- no generic observation byte arena without a concrete current need;
- no chip/AIR/executor IDs, trace heights, row counts, or record layouts.

Prefer the smallest append-only logs and derive routing, predecessor links, access
ranges, touched memory, trace sizes, and proof-specific metadata after execution.

### C versus Rust is a performance decision

Much of preflight will probably be emitted inline in generated C, following pure
and metered RVR execution, because a Rust/FFI callback for every instruction or
memory access would be expensive. This is not predetermined: prototype where
convenient and decide from generated code size, native instruction count, append
bandwidth, and end-to-end benchmarks. The semantic design must not depend on Rust
collection abstractions.

### Preserve semantic reasoning while simplifying storage

Minimizing the physical log does not justify deleting useful semantic cases from
the design. In particular, the untimed-peek reasoning must remain explicit:

- a peek does not tick or touch the memory bus;
- replay identifies its value from the already-consumed timed-event prefix inside
  the instruction, not from timestamp alone;
- peek-before-write and peek-after-write in one instruction must see different
  versions;
- repeated lookup results may be materialized after preflight without adding
  executor writes;
- genuinely non-memory, non-deterministic host inputs need a recorded observation
  if they are not materialized by logged memory writes.

Remove unnecessary storage and speculative interfaces, not correctness context.

### Avoid speculative prover abstractions

Do not invent a `LogTraceGenerator` trait, producer-role framework, transcript
header hierarchy, or similar API before implementing the executor and porting the
first real trace generators. State the ownership invariants and migrate existing
chip/filler code one consumer at a time. Let a common interface emerge only from
several concrete ports.

The low-level replay memory API is generic across address spaces. Registers,
ordinary memory, public values, and deferral memory all use the memory bus; the
address space distinguishes them. Prefer `read/write/peek(address_space, pointer)`
returning the fixed memory-bus access unit, with semantic helpers such as
`read_register` above it. Avoid the ambiguous `read_block` name because RVR also
uses “block” for basic blocks.

### Migration order must test GPU feasibility early

The implementation order should be:

1. Build and benchmark the minimal RVR preflight executor.
2. Port only the RISC-V AIR trace generators to read-only GPU replay and decide
   whether the log/index design is actually feasible and fast.
3. Complete GPU system/continuation tracegen.
4. Move to other extensions.
5. Port CPU tracegen later if it remains required.
6. Delete `RecordArena` and interpreter preflight only after all production
   consumers have migrated.

GPU tracegen is the important feasibility target. Do not build a CPU-first generic
framework or migrate every extension before measuring RISC-V GPU replay.

## Adversarial review corrections

The implementation design must also preserve these code-derived constraints:

- A terminated segment logs the fetched TERMINATE instruction and then a final
  sentinel. A suspended segment logs no synthetic instruction at the resume PC;
  its sentinel is `(resume_pc, final_timestamp)`.
- Memory predecessor timestamps are segment-local. The first event for a block has
  previous timestamp zero and executable timestamps begin at one.
- Every current memory-bus event contains exactly four field cells. This is a
  fail-closed artifact/runtime invariant, and values are packed canonically for the
  configured address-space layout.
- Do not spend `preserve_none` block arguments on log cursors. Preflight state is
  reached through `RvState::mode_state`, preserving the current hot guest-register
  budget, especially on x86-64.
- `initial_write_log` is the minimal finalized output, but its hot-path production
  is a benchmark choice. Compare a segment-local sparse seen structure with
  sequential candidates plus cold filtering. Candidate filtering must retain a
  write only when it is the block's first timed event of either kind; deduplicating
  writes alone is incorrect when a read precedes the first write.
- Basic blocks are capped at 1000 instructions, but that does not bound logical
  timestamp use. Segment planning must account for timestamp-blind metering and a
  single variable-length instruction.
- An executor error invalidates the whole transcript, but an in-place `VmState`
  may already be mutated. Prefer an owned state API that returns state only on
  success; any temporary `&mut` API is poison-on-error, not transactional.
- Rebuilding an initial Merkle view from the sparse overlay requires the complete
  final memory image and must use a separate read-only view. It must not overwrite
  the final state carried into the next segment.
- The connector owns the TERMINATE program lookup and timestamp range-check
  requests. Final public-value extraction is proved against the final segment's
  completed Merkle top tree.
- The baseline executor returns exactly the three logs. Add a concrete fourth
  observation array only if an audited current instruction consumes a
  non-deterministic, proof-visible value not recoverable from static instructions,
  immutable inputs, or logged memory. Final state may validate endpoints but cannot
  reconstruct an ordered history of nondeterministic observations.

The RISC-V GPU milestone is a real go/no-go gate: exact legacy matrices where row
order is defined and equal canonicalized lookup-request multisets otherwise, no
production `RecordArena`, and peak-to-peak memory comparison over identical phases
and inputs. Pin the workloads, input sizes, GPU/compiler configuration, warmup,
repetition count, and statistic before tuning. The total executor-plus-RISC-V-
tracegen time may regress by at most 10% on any pinned workload and must be no
slower in geometric mean than the legacy path.
