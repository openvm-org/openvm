# Compiled preflight and record-free trace generation

This document describes the current design. The original problem statement and
review context are preserved in [`context.md`](context.md).

## Objective

Execution should be owned by opcodes, not by trace-generating chips. With the
`rvr` feature, preflight runs the same compiled instruction semantics as
ordinary execution. Its extra job is narrowly defined:

> Convert serial execution over mutable random-access memory into the smallest
> replay seed from which the GPU can derive an immutable execution history.

Trace generation re-executes from that history. It does not consume chip-shaped
execution records, and the active proving path does not construct a
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

preflight
    program + mutable VM state
        -> final VM state + checkpoints + ordered residuals
```

Pure execution is the performance floor. Metering still tracks the AIR height
and touched-memory constraints needed to choose valid segment boundaries.
Preflight executes one metered segment and produces the compact input for
proving it.

Metering and preflight stop only at compiled basic-block boundaries. Preflight
must retire the instruction count selected by metering; an early termination,
over-retirement, or wrong suspended endpoint is an error.

## Pipeline

```text
                           serial, mutable

 program + segment-start state
              |
              v
       compiled preflight execution
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

### Fixed-program preparation

Generated executors and immutable program data belong to the fixed program, not
to one proof input. Under `cuda + rvr`, the ordinary `app_prover`, `prover`,
`prove-app`, and `prove-stark` APIs prepare them automatically:

```text
construct ordinary app prover
    compile metered executor
    compile preflight executor
    upload immutable replay program
             |
             v
prove(input) repeatedly
    metered execution -> preflight execution -> postflight -> tracegen -> prove
```

Preparation and proof execution have separate metric scopes. A warm proof does
not compile generated C or upload the program again. `AppProver` retains the
compiled metered and preflight execution instances and the immutable GPU
program; there is no second public prover type, execution mode, artifact
framework, or proof-record store.

Successful proofs and failures before a trace-generation session begins
leave the prepared prover reusable. A failure while that session is active is
terminal: producer lookup counts are not transactional, so the VM remains
poisoned and retries fail closed rather than proving from partially mutated
state. The caller must prepare a new prover after such an error.

## Authoritative serial output

One successful preflight execution returns:

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

### Future CPU/GPU overlap

The authoritative checkpoint format already has the right boundary for
pipelining. A checkpoint closes an interval only after generated execution has
flushed its PC, registers, timestamp, retired ordinal, and residual cursor.
Together with the preceding checkpoint, or the distinguished segment start,
that is enough for the GPU to replay the interval. It does not need an
interval-specific memory snapshot: replay emits ordered memory intents, and the
later chronology pass resolves them against the immutable segment-start image.

The current implementation is intentionally full-segment. Generated C writes
into reserved `Vec` capacity, Rust validates the complete result before setting
the vector lengths, and `expand_checkpoint_replay` performs count, emit,
chronology, and opcode indexing in one call. Reading those buffers concurrently
would be a data race, even though their allocations do not move.

The first pipelined version should introduce one concrete internal transport,
not another transcript or generic producer trait:

```text
CPU checkpoint execution
       |
       | publish immutable batches of closed intervals
       v
GPU count replay while CPU continues
       |
       | segment-end validation and final anchor
       v
exact allocation + emit + global chronology + opcode index
       |
       v
immutable GPU transcript
```

A published batch needs only its starting anchor, one or more closing anchors,
the absolute residual-base cursor, and its immutable residual slice. Publication
should use owned immutable batches or a small fixed SPSC/double buffer with an
explicit publish boundary. It must never expose a `Vec` while generated code is
still appending to it.

Streaming count first preserves the existing exact-allocation discipline. It
does not retain per-batch event logs, reserve worst-case output, require
incremental predecessor resolution, or move the memory peak into trace
generation. After the final serial boundary is validated, one emit pass writes
the exactly sized segment buffers and the existing device-wide chronology and
indexing passes run unchanged. Any overlapped count work is speculative and is
discarded if serial execution or endpoint validation fails.

Streaming emit can be considered separately only if a full-workload benchmark
shows that count overlap matters and device chunks can be assembled without a
second full-size buffer. Streaming chronology or AIR trace generation is a
different, substantially more invasive optimization and is not implied by this
boundary.

This overlap is not the only reason to keep expansion separate from serial
execution. Even without overlap, separation keeps proof-shaped work off the CPU
hot path, lets checkpoint intervals expand in parallel, produces one shared
read-only history for all AIRs, and keeps the authoritative serial output
minimal.

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
2. executes preflight to the metered boundary;
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

The active compiled preflight executor, GPU expansion, standard SDK GPU tracegen,
and continuation proving path do not construct a `RecordArena`.

The direct full-log RVR preflight executor remains available as a correctness
oracle for differential and negative tests. It is not a second production
preflight contract. Restricting that oracle to test utilities can happen after
its integration callers have an appropriate feature boundary; deleting its
coverage or routing production through it would be a regression.

`RecordArena` still exists for legacy interpreter preflight, legacy/default GPU
builders, CPU trace generation, and tests that have not moved to read-only
replay. Removing it from those APIs is a separate repository-wide migration.
The compiled path must not reintroduce a record adapter merely to make that
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

The clean exact-source standalone execution comparison reports:

```text
mode        median execution   generated-C compilation
pure          738.586 ms              87.004 s
metered      1390.368 ms             153.896 s
preflight   1369.164 ms             145.622 s
```

All three modes retired 501,243,291 guest instructions and passed the exact
endpoint, segment, and output checks. Checkpoint execution was 1.5% faster than
metered and about 1.85 times pure execution. Generated-C compilation is
one-time fixed-program preparation and remains reported separately from proof
execution. This standalone harness and the proof runner below are separate
builds and report instruction totals that differ by 3,627; performance
comparisons are only between matched modes within each table.

Focused pairing tests kept proving as the peak phase: BN254 used about 355 MiB
during trace generation versus 1.5 GiB during proving, and BLS12-381 used about
560 MiB versus 2.2 GiB. These small tests validate lifetimes but do not replace
the full-workload GPU-memory gate.

The exact-source legacy and compiled-preflight Reth proofs used the same
release binary, input, 15 GiB segmentation estimate, and PID-scoped 0.2-second
GPU sampler. The compiled preflight proof verified all 55 segments and the
expected output:

```text
OpenVM       13cddd7cefb2cccbe52ed5864d874403d436e9cf
stark-backend be2b6983cbd70976b37acbb72ceb1b3593dc67ae
openvm-eth   a8f6ad8a61ec7f874cab458ff3c3caf3d7d90a34
rvr-openvm   604ad55aa9cd7a5a9639f45ec7c27a22915b48a0
input SHA-256 97097c091120b2c09657917d4d3b95c61ec2e3dd25b3b210414f087d80c5a898
```

```text
phase                               legacy records   compiled preflight
guest instructions / segments      501,246,918 / 55   501,246,918 / 55
one-time executor preparation          133.576 s          278.097 s
reusable app proof                     103.075 s           68.695 s
metered execution                        1.457 s            1.472 s
serial preflight                        31.287 s            1.385 s
GPU replay expansion                         -             2.907 s
trace generation                         5.312 s            2.709 s
proving excluding tracegen              62.605 s           58.046 s
sum of the timed phases above          100.661 s           66.519 s
warm process wall                      122.714 s           84.976 s

postflight allocation peak                                2.518 GiB
preflight tracegen allocation peak                        5.907 GiB
legacy proving allocation peak          15.069 GiB
compiled proving allocation peak                          15.077 GiB
PID-scoped process peak                 16,428 MiB       16,440 MiB
PID-scoped process peak delta                                +12 MiB
```

Expansion and trace generation remain well below proving, and replay buffers
are not retained at the proving peak. Serial preflight is 22.6 times faster,
trace generation is 49.0% faster, the timed phase sum is 33.9% faster, and the
reusable app proof is 33.4% faster. The prepared app-proof value is a direct
metric. The legacy reusable proof and both warm process values subtract their
one-time preparation from otherwise exact runs because the legacy benchmark
does not expose a prepared prover.

The identical PID-scoped process peak increased by 12 MiB, about 0.07%, and
remained in proving rather than moving to expansion or trace generation. The
allocator's proving peak increased by about 8 MiB. Both paths exceed a strict
15.0 GiB process cap because the 15 GiB segmentation value estimates proof
buffers rather than the CUDA context and allocator pool. Packing that requires
a lower segmentation limit, but postflight replay is not responsible for the
existing proving peak.

The definitive prepared run verified every segment, endpoint continuity, the
final public-values Merkle proof, and the output
`b0c6920a15b5f11db176fcd1b22754fe845f9f5b24a245f1c67b997f353f3878`
followed by the expected zero half. The preparation and proof spans were
siblings: `compile_metered` took 133.412 seconds,
`compile_preflight` took 144.583 seconds, immutable program upload
took 50 milliseconds, and no compilation or upload occurred inside the
68.695-second app-proof span. The checkpoint transcript contained 962,366
checkpoints and 129,268,505 residual words, or 1,288,212,664 logical payload
bytes and 2.57 bytes per guest instruction.

One-time compilation remains an optimization target, not a proof-time cost.
The clean host release build used the repository's existing fat-LTO profile and
took 5 minutes after the independently built guest. Generated executors use
thin LTO. The exact standalone comparison above remains the like-for-like
compile-time gate for pure, metered, and preflight execution.

## Follow-up work

1. Reduce cold generated-C compilation only when a full-workload comparison
   preserves runtime performance and does not increase Rust/CUDA build cost.
2. Instrument checkpoint-batch count and transfer time, then prototype
   closed-interval count overlap only if the whole Reth proof improves without
   increasing expansion or tracegen peak GPU memory.
3. Decide separately whether legacy CPU/interpreter support is migrated far
   enough to remove `RecordArena` from shared builder traits. The compiled
   proving path already avoids it.

## Performance and maintainability gates

Runtime performance, generated-code compile time, Rust/CUDA compile time, peak
GPU memory, and reviewability are all acceptance criteria.

Track at least:

- checkpoint and residual bytes per guest instruction;
- native instructions per guest instruction;
- pure, metered, and preflight execution time;
- generated-C and Rust/CUDA compilation time;
- upload, postflight replay, sorting/indexing, and trace-kernel time;
- segment proof time and total proof time;
- live and reserved GPU memory for postflight, tracegen, and proving.

The production metrics surface stays phase-level and low-cardinality:

- `prepare_preflight_time_ms`, containing the one-time preparation only;
- the existing `compile_metered_time_ms` plus
  `compile_preflight_time_ms`, both attributed to preparation;
- `upload_preflight_program_time_ms`, attributed to preparation;
- `app_prove_time_ms`, excluding preparation;
- `execute_preflight_time_ms`, attributed only by the existing
  segment scope;
- `execute_preflight_insns` and
  `execute_preflight_insn_mi/s`, emitted once per completed proof;
- `execute_preflight_checkpoints`,
  `execute_preflight_residuals`, and
  `execute_preflight_transcript_bytes`, emitted once per completed
  proof;
- `postflight_time_ms` and its four fixed subphases, attributed only by
  segment;
- the existing `trace_gen`, `system_trace_gen`, `executor_trace_gen`, and
  proving metrics.

The new path reuses the existing low-cardinality `opcode_count` and
`single_trace_gen` metrics. There are no dynamic-PC or generated-kernel labels.
The profiler checks the proof-level preflight instruction total against metered
execution when both metrics are present.

Native instructions per guest instruction, generated source/object size, host
compiler RSS, and sampled GPU process memory are offline benchmark/profiler
measurements rather than production metrics. The standalone `rvr-openvm`
benchmark reports host instruction counters beside guest instructions so the
ratio is derived without adding high-volume runtime instrumentation.

An optimization is accepted only when it improves the real workload without
weakening the execution contract or extending the serial transcript. In
particular:

- no per-chip buffers or proof-shaped metadata on the serial hot path;
- no compatibility records in the active preflight path;
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
