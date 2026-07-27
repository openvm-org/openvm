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
execution records.

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
                           postflight
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
             |
             v
first prove(input)
    one-time preparation:
        compile metered executor
        compile preflight executor
        upload immutable replay program
    reusable proof:
        metered -> preflight -> postflight -> tracegen -> prove
             |
             v
later prove(input)
    reuse preparation:
        metered -> preflight -> postflight -> tracegen -> prove
```

Preparation and proof execution have separate metric scopes. A warm proof does
not compile generated C or upload the program again. `AppProver` retains the
compiled metered and preflight execution instances and the immutable GPU
program; there is no second public prover type, execution mode, artifact
framework, or proof-record store.

App proving-key generation has not moved between CPU and GPU. As on
`develop-v2.1.0`, `AppProvingKey::keygen` uses the CPU engine to construct
backend-independent AIR and proving-key metadata once, and the SDK caches the
result. `new_local_prover` then creates the selected proving engine and
transports that key to its device before committing the program. With the CUDA
SDK builder, trace generation and proving remain GPU work. Compiled executor
preparation is a separate fixed-program cost and is not key generation.

Successful proofs and failures before a trace-generation session begins
leave the fixed-program prover reusable. A failure while that session is active is
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

Tests can copy and deliberately corrupt the derived program and memory logs, and
the chronology tests retain an independent CPU index oracle. There is no second
serial executor that produces a competing full-log preflight contract.

## Derived read-only history

Postflight performs two GPU passes over the same checkpoint intervals:

1. Count the exact number of memory and field-memory events. The program-log
   length is the retired instruction count plus its final boundary.
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
flushed its timestamp, retired ordinal, and residual cursor, then captured its
PC and registers.
Together with the preceding checkpoint, or the distinguished segment start,
that is enough for the GPU to replay the interval. It does not need an
interval-specific memory snapshot: replay emits ordered memory intents, and the
later chronology pass resolves them against the immutable segment-start image.

The current implementation is intentionally full-segment. Generated C writes
into reserved `Vec` capacity. Rust validates the returned ABI, bounds, and
timestamp before setting the vector lengths, then validates checkpoint
semantics. `postflight` performs count, emit, chronology, and
opcode indexing in one call. Reading those buffers concurrently would be a data
race, even though their allocations do not move.

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
the absolute residual-base cursor, its immutable residual slice, and whether
the last boundary is interior or the segment endpoint. Publication should use
owned immutable batches or a small fixed SPSC/double buffer with an explicit
publish boundary. It must never expose a `Vec` while generated code is still
appending to it.

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

### Hint-store alignment contract

This branch intentionally changes one guest-visible error case. The legacy
interpreter accepted a hint-store destination as an arbitrary byte address,
while the AIR exposes each proof-visible hint write as one aligned eight-byte
memory block. Compiled execution correctly rejected the unaligned form, which
initially created an executor divergence.

The interpreter and compiled executors now both require eight-byte alignment
and reject other destinations. This is a contract correction, not a preflight
optimization, and should be reviewed as such. Differential tests pin the shared
behavior.

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

## Execution-history boundary

Interpreted preflight appends the generic execution history directly. Compiled
preflight emits checkpoints and residuals, which postflight expands into the
same history. Standard SDK tracegen and continuation proving begin only after
both paths have converged on that validated read-only representation.

Negative tests construct the generic history directly, while chronology tests
retain an independent CPU oracle. Neither introduces a second execution mode or
production contract.

CPU trace generation replays the validated host history. GPU trace generation
consumes the uploaded immutable history and its postflight indexes. Tests use
the same paths rather than retaining a second execution contract.

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

The definitive Reth comparison used OpenVM `c215dbf3ac`, openvm-eth
`3ab635e6`, block `24001988`, the same cached RPC input and guest ELF, the same
CUDA runner class, and the same proof configuration. The only execution-mode
difference was compiled preflight versus the legacy interpreter. Both runs
retired exactly 664,625,965 guest instructions, produced 74 segments, verified
the app proof, and reached the same GPU-memory peak.

```text
phase                                      interpreter   compiled preflight
guest instructions / segments        664,625,965 / 74   664,625,965 / 74
one-time executor preparation                0.041 s          744.256 s
  compile metered                            0.041 s          347.492 s
  compile preflight                                -          396.724 s
  immutable program upload                         -            0.020 s
reusable app-prove span                     65.061 s           48.817 s
sum of segment proof spans                  61.529 s           47.692 s
metered execution                            3.391 s            1.059 s
serial preflight                            14.518 s            1.040 s
postflight                                       -             1.644 s
trace generation                             2.376 s            1.095 s
initial-memory upload                        1.206 s            1.215 s
backend proving excluding tracegen          43.313 s           42.538 s
other orchestration and timer rounding       0.257 s            0.226 s
peak GPU memory                              15.80 GB           15.80 GB
```

Compilation is deliberately outside the reusable app-proof span and happens
once per prepared fixed-program prover. It is not paid per input or per
segment. The CI runner compiled more slowly than the standalone CUDA host, so
the standalone exact-mode table remains the better generated-code compilation
comparison.

Within the reusable proof, serial preflight is 14.0 times faster and trace
generation is 53.9% faster. The new 1.644-second postflight phase does not erase
those gains: the app-prove span is 25.0% faster and the sum of segment proof
spans is 22.5% faster. Frontend work measured as app-prove time minus backend
proving fell from 21.748 seconds to 6.279 seconds, a 71.1% reduction.

Postflight peaked at 2.79 GB and trace generation at 5.96 GB, while proving
remained the peak phase at 14.73 GB of attributed allocation. The process peak
reported by `nvidia-smi` was exactly 15.80 GB in both runs, so replay scratch
did not move the global peak out of proving.

The transcript contained 1,274,139 checkpoints, 180,895,403 residual words,
and 1,783,535,920 payload bytes, or 2.68 bytes per guest instruction.

The matching workflow runs are:

- compiled preflight:
  [30197418677](https://github.com/axiom-crypto/openvm-eth/actions/runs/30197418677);
- legacy interpreter:
  [30197836673](https://github.com/axiom-crypto/openvm-eth/actions/runs/30197836673).

[PR 3020](https://github.com/openvm-org/openvm/pull/3020) was also
[run](https://github.com/axiom-crypto/openvm-eth/actions/runs/30197475225)
on the same block as an external reference. It retired 664,649,751
instructions, 23,786 more than the matched pair, so it is not used as the
authoritative before/after baseline. Its 74 segment proof reported 7.919
seconds of preflight, 13.497 seconds of trace generation, 66.101 seconds across
app segment proof spans, and 16.18 GB peak GPU memory.

For a historical comparison, the strict pre-change baseline is the merge base,
OpenVM `e2bced3b`, with compiled RVR metering and legacy per-chip preflight.
The benchmarked revision was `00afc086`. Local CUDA proofs used byte-identical guest
ELFs, cached block inputs, and VM configuration in each pair. All six proofs
verified, and each pair produced the same number of segments.

```text
block / phase                         strict pre-change     current
23992138
  guest instructions / segments    501,255,045 / 54   501,244,565 / 54
  one-time executor preparation          130.450 s          277.575 s
  metered execution                        1.371 s            1.542 s
  preflight                               30.403 s            1.443 s
  postflight                                   -             2.181 s
  trace generation                         5.141 s            2.622 s
  initial-memory upload                    1.839 s            1.838 s
  backend proving excluding tracegen      59.725 s           54.340 s
  sum of segment proof spans              97.195 s           62.561 s
  peak GPU memory                         17,083 MiB         17,055 MiB

24846099
  guest instructions / segments    532,124,531 / 51   532,120,058 / 51
  one-time executor preparation          129.485 s          281.044 s
  metered execution                        3.268 s            3.342 s
  preflight                               34.646 s            3.251 s
  postflight                                   -             2.291 s
  trace generation                        19.707 s            6.177 s
  initial-memory upload                    0.886 s            0.904 s
  backend proving excluding tracegen      53.366 s           50.243 s
  sum of segment proof spans             108.688 s           62.990 s
  peak GPU memory                         17,053 MiB         17,029 MiB

25563139
  guest instructions / segments    514,887,163 / 50   514,877,771 / 50
  one-time executor preparation          129.501 s            9.834 s*
  metered execution                        4.536 s            4.598 s
  preflight                               35.211 s            4.538 s
  postflight                                   -             2.218 s
  trace generation                        22.687 s            9.859 s
  initial-memory upload                    0.955 s            0.974 s
  backend proving excluding tracegen      55.679 s           52.603 s
  sum of segment proof spans             114.609 s           70.317 s
  peak GPU memory                         17,049 MiB         17,029 MiB
```

`*` The current run reused the cross-process native artifact prepared by an
earlier block. The two cold current runs spent about 280 seconds compiling
metered and preflight executors, versus about 130 seconds for the historical
metered executor alone. Compilation remains outside the segment proof spans.

The historical executor retired between 4,473 and 10,480 more instructions
despite identical inputs and guest code, a difference of 0.0008% to 0.0021%.
The branch includes RVR correctness and hardening changes in addition to the
preflight pipeline, so this historical table is not an execution-only
experiment. The exact-instruction interpreter comparison above remains the
isolated preflight measurement.

Across the historical workloads, preflight is 7.8 to 21.1 times faster and
trace generation is 2.0 to 3.2 times faster. Including postflight and
initial-memory upload, measured frontend work fell by 70.1% to 78.4%. The sum
of segment proof spans fell by 35.6% to 42.0%, and proving remained the peak
GPU-memory phase.

Matched CI runs confirm the same result on the default block and the two
extension-heavy blocks:

```text
block / phase                         strict pre-change     current
24001988
  guest instructions / segments    664,708,221 / 73   664,680,064 / 73
  one-time executor preparation          338.451 s          743.051 s
  preflight                               14.881 s            1.043 s
  postflight                                   -             1.643 s
  trace generation                         2.370 s            1.101 s
  sum of segment proof spans              61.718 s           48.097 s
  peak GPU memory                          15.79 GB           15.83 GB

24846099
  guest instructions / segments    532,181,524 / 51   532,150,076 / 51
  one-time executor preparation          336.385 s          744.411 s
  preflight                               14.061 s            2.678 s
  postflight                                   -             1.396 s
  trace generation                        18.965 s            4.310 s
  sum of segment proof spans              65.811 s           41.361 s
  peak GPU memory                          15.89 GB           15.87 GB

25563139
  guest instructions / segments    514,932,464 / 50   514,898,244 / 50
  one-time executor preparation          339.524 s          738.208 s
  preflight                               15.036 s            3.798 s
  postflight                                   -             1.382 s
  trace generation                        24.567 s            6.764 s
  sum of segment proof spans              75.500 s           47.818 s
  peak GPU memory                          15.89 GB           15.96 GB
```

The two runs in each CI pair used the same workflow inputs and restored RPC
input cache. Each branch built its own guest artifact, so the local
byte-identical-ELF table above is the controlled comparison. In the
end-to-end CI comparison, the historical revision retired 0.0042% to 0.0066%
more instructions while segment counts stayed equal. Frontend work including
postflight and initial-memory upload fell by 69.2% to 73.8%. Segment proof
spans fell by 22.1% on the default block and 36.7% to 37.2% on the
extension-heavy blocks.

The historical workflows are:

- block `24001988`:
  [30201287271](https://github.com/axiom-crypto/openvm-eth/actions/runs/30201287271);
- block `24846099`:
  [30201286921](https://github.com/axiom-crypto/openvm-eth/actions/runs/30201286921);
- block `25563139`:
  [30201287142](https://github.com/axiom-crypto/openvm-eth/actions/runs/30201287142).

The matching current workflows are:

- block `24001988`:
  [30198644614](https://github.com/axiom-crypto/openvm-eth/actions/runs/30198644614);
- block `24846099`:
  [30199862246](https://github.com/axiom-crypto/openvm-eth/actions/runs/30199862246);
- block `25563139`:
  [30199862810](https://github.com/axiom-crypto/openvm-eth/actions/runs/30199862810).

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

## Review stack

The implementation is large because every active AIR must consume the same
read-only history. Review should be split by dependency and ownership, not by
arbitrary line count. A nine-change stack keeps each boundary concrete:

1. generic preflight history, interpreter logging, host postflight indexing,
   and system boundary semantics;
2. CPU history trace generation for system, RISC-V, and extensions;
3. compiled preflight execution, checkpoints, residuals, and differential
   execution tests;
4. generic GPU postflight, chronology, system traces, continuation state, and
   shared validation;
5. RV64 GPU replay, grouped into control/ALU, memory, M, and IO changes;
6. bigint, modular arithmetic, field expressions, and ECC GPU replay;
7. Keccak-256, SHA-2, deferral, and pairing GPU replay;
8. the ordinary SDK proving path and low-cardinality metrics;
9. final RecordArena/legacy purge, fixture consolidation, performance evidence,
   and documentation.

Small correctness prerequisites that are independently reviewable should land
before the relevant layer: the aligned four-cell memory contract, typed
out-of-bounds failures, x0 equality behavior, repeated-curve fixtures,
field-expression serialization, malformed pairing-pointer rejection, and GPU
peak-memory instrumentation. The history can be rewritten into this stack
before review; it need not be rewritten while performance and correctness work
is still active.

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
- `execute_preflight_insns`, emitted once per completed segment, and
  `execute_preflight_insn_mi/s`, emitted once for the completed proof;
- `execute_preflight_intervals`,
  `execute_preflight_residuals`, and
  `execute_preflight_transcript_bytes`, emitted once for the completed proof;
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
