# RVR checkpoint preflight

This document describes the current design. The original problem statement and
review context are preserved in [`context.md`](context.md).

## Objective

Execution should be owned by opcodes, not by trace-generating chips. Preflight
therefore runs the same compiled RVR instruction semantics as ordinary
execution. Its extra job is narrowly defined:

> Convert serial execution over mutable random-access memory into the smallest
> replay seed from which the GPU can derive an immutable execution history.

Trace generation re-executes from that history. It does not consume chip-shaped
execution records, and the active checkpoint proving path does not construct a
`RecordArena`.

## Three execution modes

The modes share opcode semantics and generated basic blocks. They differ only in
the bookkeeping required by their caller:

```text
pure
    program + mutable VM state
        -> final VM state

metered
    program + mutable VM state
        -> final VM state + per-AIR segment plan + replay-size bounds

checkpoint preflight
    program + mutable VM state
        -> final VM state + checkpoints + ordered residuals
```

Pure execution is the performance floor. Metering still tracks the AIR height
and touched-memory constraints needed to choose valid segment boundaries.
Checkpoint preflight executes one exact metered segment and produces the compact
input for proving it.

Metering and checkpoint preflight stop only at RVR basic-block boundaries. The
production continuation entry point requires the checkpoint executor to retire
the exact instruction count selected by metering; an early termination,
over-retirement, or wrong suspended endpoint is an error.

## Pipeline

```text
                           serial, mutable

 program + segment-start state
              |
              v
       compiled RVR execution
              |
              +----> final VM state
              |
              `----> checkpoints + ordered residuals
                              |
                              v
                         GPU expansion
                  re-execute checkpoint intervals
                              |
                              v
                derived immutable segment history
              program log + memory log + chronology
                    + opcode-partitioned indexes
                              |
             +----------------+----------------+
             |                |                |
             v                v                v
        system AIRs       RV64 AIRs      extension AIRs
             \                |                /
              `---------------+---------------'
                              |
                              v
                    trace matrices + lookups
                              |
          drop segment replay buffers before proving
```

The first GPU pass is sometimes called expansion or postflight. It is not a
second authoritative record format. It is a derived, device-resident view whose
lifetime ends before the proving/GKR memory peak.

### Prepared prover lifecycle

Generated executors and immutable program data belong to the fixed program, not
to one proof input. The SDK therefore exposes an explicit prepared checkpoint
prover:

```text
prepare once
    compile metered executor
    compile checkpoint executor
    upload immutable replay program
             |
             v
prove(input) repeatedly
    metered execution -> checkpoint execution -> GPU replay -> prove
```

Preparation and proof execution have separate metric scopes. A warm proof does
not compile generated C or upload the program again. The prepared prover borrows
the SDK's existing executor, following the same ownership model as the compiled
pure and metered SDK executors; it does not add a generic artifact framework or
store proof records. The caller names the ordinary app prover before preparing
it, so setup and proof metrics share the same low-cardinality app group.

## Authoritative serial output

One successful checkpoint execution returns:

```text
checkpoints:
    [(pc, timestamp, retired, residual_cursor, x1..x31), ...]

residuals:
    [u64, ...]
```

The initial and final execution boundaries, endpoint, streams, and mutable
memory state already belong to the execution call. They stay beside the
transcript instead of being copied into it.

There is deliberately no:

- schema version or transcript header;
- program or configuration digest;
- per-instruction step array;
- serial program or memory event log;
- seed or observation arena;
- AIR, chip, executor, or trace-height identifier;
- extension-specific execution record.

Checkpoints are periodic architectural anchors placed at existing RVR block
boundaries. A checkpoint interval can be expanded independently because it
starts with the complete register file, PC, timestamp, retired count, and
residual cursor.

Residuals contain only ordered values that deterministic replay cannot recover
from the program, checkpoint state, and immutable segment-start memory image.
Examples include load results, host-advice-dependent control values, and
extension output postimages. Static schedules determine how many residuals an
instruction consumes, so residuals need no per-value tag.

Capacity is bounded before execution. Arithmetic overflow, exhausted checkpoint
or residual capacity, a malformed callback result, and a transcript cursor
mismatch fail the segment rather than returning a partial transcript.

The direct-event RVR preflight implementation remains a differential oracle. It
logs semantic program and memory events directly and is useful for tests, but it
is not the production proving architecture.

## Derived read-only history

GPU expansion performs two passes over the same checkpoint intervals:

1. Count the exact number of program, memory, and field-memory events.
2. Allocate exact output sizes and emit those events.

Each interval is sequential internally because instructions share PC,
registers, timestamp, and the residual cursor. Intervals are independent and
expand in parallel. After expansion, trace rows are partitioned by opcode and
generated in parallel from immutable buffers.

The derived segment view contains:

- a program log ordered by executed-instruction ordinal, with the instruction's
  PC and starting timestamp plus one final boundary;
- a memory log ordered by global timed-event ordinal;
- full-width field values where a four-cell value does not fit in the compact
  event;
- predecessor indexes for timed memory accesses;
- first-write initial values and final touched-block values;
- an opcode-partitioned step index and program-row frequencies.

Registers and ordinary memory use the same logical memory model. An event is
identified by address space, aligned block pointer, timestamp, kind, and value.
The current memory bus has a uniform four-field-cell block width; program upload
rejects an incompatible configuration.

Memory chronology groups events by `(address_space, block_pointer)` while
retaining their global event order. A sort and segmented scans reconstruct each
event's complete post-value, predecessor timestamp, predecessor value, and the
final value of every touched block from:

- the immutable segment-start memory image;
- the ordered read/write events;
- write masks and output postimages.

The resulting logs are read-only inputs. AIR-specific kernels may select and
re-execute the rows they own, but they do not mutate VM state and do not create
another compatibility record.

## Timestamp and access semantics

Logical timestamp is not the number of emitted memory events. AIR schedules
reserve clock slots even when no memory-bus interaction occurs.

A timed read or write:

- consumes its scheduled timestamp slot;
- emits one canonical memory event when enabled;
- participates in predecessor and touched-memory construction.

The important clock-only cases are:

- a disabled destination write or a write to `x0`;
- the unused second-block slot of a non-crossing load or store;
- HintStore's fixed scheduling gaps;
- Phantom instructions.

Immediate operands do not consume timestamp slots. They are carried through the
execution bus. For example, an immediate ALU instruction consumes its register
read and destination-write slots, a branch consumes its two register reads, and
a load/store offset is constrained as an instruction operand.

The replay schedule must advance over clock-only slots explicitly. It must never
invent a fake memory event merely to make timestamps contiguous.

The first timed event for a block in a segment has predecessor timestamp zero.
Predecessor links are segment-local and never cross a continuation boundary.

## Peeks and advice

A peek appends nothing, consumes no timestamp, and does not mark a block touched.
Replay resolves it from the memory version immediately after the timed-event
prefix already consumed at that logical point.

This prefix, rather than timestamp alone, distinguishes a peek before a write
from a peek after a write within the same instruction. Several peeks may occur
at one timestamp without ambiguity because replay executes their local order.

A proof-visible memory read must be timed. A value used only through a peek is
uncommitted execution advice. If its effect becomes proof-visible, replay sees
that effect through a later timed access or through the ordered residual chosen
by the opcode's execution contract.

Host hints, randomness, deferral callbacks, output, and other side effects run
once during serial execution. Trace generation does not repeat host side
effects. It replays their materialized effects from residuals and timed memory
writes.

## Replay ownership

The immutable program upload contains native RV64 opcode metadata and a compact
registry of extension-owned access schedules. A schedule describes only the
ordered register accesses, bounded memory spans, clock gaps, residual/static
write sources, and simple PC or register effects needed to expand the common
history.

This registry is program metadata, not serial transcript data. Program upload
checks that opcode ownership is disjoint and that every executed opcode has a
producer.

After expansion, concrete coordinators visit the existing VM inventory:

- the RV64 coordinator generates base arithmetic, control-flow, load/store,
  multiply/divide, HintStore, and Phantom traces;
- extension coordinators claim their own opcodes and generate their traces;
- the system path generates Program, Connector, memory boundary/Merkle,
  persistent-boundary, and shared periphery requests from the same history.

Exactly one primary producer emits an executed instruction's execution/program
interaction. Auxiliary producers may consume the same instruction without
duplicating that interaction, which permits one opcode to feed several traces.

There is intentionally no generic `LogTraceGenerator` trait. The common
abstraction is the read-only history and its indexes. AIR-specific generation
stays in the existing concrete chip inventory, where row layout and lookup
ownership are visible.

All kernels share one sticky replay error. The coordinator synchronizes once,
checks that error, and rejects the whole proving context on any malformed
pointer, timestamp, cursor, opcode, canonical value, or unsupported schedule.

## Continuations and dirty pages

Metering selects segment boundaries. For each segment the proving coordinator:

1. uploads the immutable segment-start memory image before execution mutates it;
2. executes checkpoint preflight to the exact metered boundary;
3. expands checkpoints and residuals on the GPU;
4. generates all system and extension traces from the derived history;
5. synchronizes and releases transcript, chronology, sort, and replay-index
   buffers;
6. proves the segment;
7. carries only the returned final `VmState` into the next segment.

Dirty-page bitsets for main memory, public values, and deferral memory are sparse
host-to-device transfer metadata. They are updated by validated writes,
including callback materialization, but are not transcript fields and are not
consumed by replay.

Initial Merkle state belongs to the continuation coordinator. Reconstructing the
initial touched-memory view must never overwrite the final memory state carried
to the next segment. Final public values are extracted using the final segment's
completed memory top tree.

## RecordArena boundary

The active RVR checkpoint executor, GPU expansion, standard SDK GPU tracegen,
and continuation proving path do not construct a `RecordArena`.

`RecordArena` still exists for legacy interpreter preflight, legacy/default GPU
builders, CPU trace generation, and tests that have not moved to checkpoint
replay. Removing it from those APIs is a separate repository-wide migration.
The checkpoint path must not reintroduce a record adapter merely to make that
cleanup look complete.

## Current implementation status

The following are implemented with focused GPU prove-and-verify coverage:

- system traces and complete RV64IM replay, including branches, all load/store
  widths and crossings, HintStore, Phantom, public values, suspension, and
  continuation state;
- Keccak-256, SHA-2, and Int256;
- modular arithmetic, Fp2, and Weierstrass/ECC field-expression traces;
- deferral CALL/OUTPUT and its auxiliary traces;
- standard SDK composition with fail-closed opcode ownership;
- genuine BN254 and BLS12-381 pairing-hint executions across checkpoint
  boundaries, including advice generation followed by HintStore
  materialization.

The pinned full-workload execution comparison currently reports:

```text
mode        median execution   generated-C compilation
pure          766.924 ms              88.986 s
metered      1610.067 ms             166.172 s
checkpoint   1567.966 ms             154.334 s
```

Metered and checkpoint execution both retired 620,281,236 guest instructions
with the same 63 segment boundaries and output hash. Checkpoint execution was
2.6% faster than metered and about 2.04 times pure execution. Generated-C
compilation is one-time fixed-program preparation and remains reported
separately from proof execution.

Focused pairing tests kept proving as the peak phase: BN254 used about 355 MiB
during trace generation versus 1.5 GiB during proving, and BLS12-381 used about
560 MiB versus 2.2 GiB. These small tests validate lifetimes but do not replace
the full-workload GPU-memory gate.

The exact-source legacy and checkpoint 63-segment Reth proofs both completed:

```text
phase                              legacy records   checkpoint replay
guest instructions / segments     620,281,236 / 63 620,281,236 / 63
serial preflight                       38.202 s          1.654 s
GPU replay expansion                        -           3.517 s
trace generation                        6.596 s          3.023 s
proving excluding tracegen              74.672 s         71.790 s
total segment work                     121.893 s         82.488 s
estimated warm runner wall             148.121 s        105.446 s

checkpoint expansion allocation peak                      2.613 GiB
checkpoint tracegen allocation peak                       5.901 GiB
checkpoint proving allocation peak                       14.882 GiB
checkpoint sampled process peak                          15.482 GiB
```

Expansion and trace generation remain well below proving, and replay buffers
are not retained at the proving peak. Serial preflight is 23.1 times faster,
trace generation is 54.2% faster, and total segment work is 32.3% faster. The
warm runner values are currently derived by subtracting generated compilation
from otherwise exact runs. The prepared-prover acceptance run replaces that
estimate with a direct measurement.

One-time preparation remains visible: legacy metered compilation took 138.596
seconds; the checkpoint path took 167.763 seconds for metered compilation and
198.883 seconds for checkpoint compilation. The host fat-LTO build increased
from 10m34.35s to 11m22.62s. These are setup and build costs, not per-proof
costs, but both remain compile-time acceptance metrics.

## Remaining acceptance work

Before calling the cutover complete:

1. Run the prepared 63-segment Reth mode and verify that the prepared-proof
   metric contains no generated compilation or immutable program upload.
   Re-check proofs, public values, output hash, endpoint continuity, and
   phase-specific GPU memory.
2. Repeat cold generated-code compilation enough to separate stable source-size
   cost from machine noise. Keep runtime optimization unchanged unless a
   full-workload comparison justifies a mode-specific alternative.
3. Rebase on the latest `origin/develop-v2.1.0` and finish the aggregate diff
   review for simple ownership, module-scoped imports, shared replay helpers,
   typed malformed-input errors, and consistent pointer bounds.
4. Decide separately whether legacy CPU/interpreter support is migrated far
   enough to remove `RecordArena` from shared builder traits.

## Performance and maintainability gates

Runtime performance, generated-code compile time, Rust/CUDA compile time, peak
GPU memory, and reviewability are all acceptance criteria.

Track at least:

- checkpoint and residual bytes per guest instruction;
- native instructions per guest instruction;
- pure, metered, and checkpoint execution time;
- generated-C and Rust/CUDA compilation time;
- upload, expansion, sorting/indexing, and trace-kernel time;
- segment proof time and total proof time;
- live and reserved GPU memory for expansion, tracegen, and proving.

The metrics surface stays phase-level and low-cardinality:

- `prepare_rvr_checkpoint_time_ms`, containing the one-time preparation only;
- the existing `compile_metered_time_ms` plus
  `compile_checkpoint_preflight_time_ms`, both attributed to preparation;
- `upload_checkpoint_program_time_ms`, attributed to preparation;
- `app_prove_rvr_checkpoint_time_ms`, excluding preparation;
- `execute_checkpoint_preflight_time_ms`, attributed only by the existing
  segment scope;
- `execute_checkpoint_preflight_insns` and
  `execute_checkpoint_preflight_insn_mi/s`, emitted once per completed proof;
- `expand_checkpoint_replay_time_ms`, attributed only by segment;
- the existing `trace_gen`, `system_trace_gen`, `executor_trace_gen`, and
  proving metrics.

There are no per-opcode, per-kernel, or dynamic instruction labels. The
profiler checks the proof-level checkpoint instruction total against metered
execution when both metrics are present.

An optimization is accepted only when it improves the real workload without
weakening the execution contract or extending the serial transcript. In
particular:

- no per-chip buffers or proof-shaped metadata on the serial hot path;
- no compatibility records in the active checkpoint path;
- no unbounded tracegen scratch allocation;
- no retained replay allocation at the proving peak;
- no large generic framework whose only purpose is hiding concrete AIR logic;
- no duplicated opcode semantics when one small schedule or helper expresses
  the shared invariant.

The stable mental model is:

```text
mutable execution
    -> minimal checkpoints + ordered residuals
    -> derived immutable history
    -> parallel concrete trace generation
```
