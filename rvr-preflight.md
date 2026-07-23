# RVR preflight without execution records

## Status

This document proposes a replacement for interpreter preflight, `RecordArena`, and
chip-shaped execution records. It deliberately separates two questions:

1. What happened during serial execution?
2. Which traces should be generated from what happened?

The first implementation milestone is only an RVR preflight executor which answers
the first question. Existing proof generation remains in place until a later
migration milestone.

The proposal is based on the current RVR and preflight implementations, including
the system AIRs and continuation path. It is a semantic design: the physical log
layout is an explicit benchmark decision, not part of the proof interface.

## Decision

Preflight is normal RVR execution with an append-only transcript of OpenVM's
**logical** instruction and memory operations. It does not know about AIR IDs,
chips, trace heights, record layouts, or prover backends.

After a segment finishes, an indexing step freezes the transcript and derives the
views needed by trace generators. CPU or GPU trace generators then replay each
instruction against a read-only view of that transcript. Exactly one trace must
emit the instruction's execution/program interactions; additional traces may use
the same instruction for one-to-many cases such as Keccak.

```text
                serial, mutable                    parallel, immutable

 program ──► RVR execution ──► preflight transcript ──► transcript indexes
                    │                                      │
                    └──► final VM state                    ├──► system traces
                                                           ├──► chip trace A
                                                           ├──► chip trace B
                                                           └──► chip trace N
```

The authoritative preflight output contains no per-chip records. `RecordArena` is
removed after all trace generators consume the transcript directly.

## Why a PC log and a write log are not enough

An instruction ordinal and the memory timestamp are different axes:

- one instruction can make several timed memory accesses;
- a disabled or `x0` destination consumes a timestamp without a write;
- a non-crossing load or store can reserve the timestamp of a second block access;
- hint-store consumes deliberate clock-only slots between variable-row writes;
- a peek observes memory without incrementing the timestamp.

Immediate values in the current RV64 AIRs are execution-bus operands and do not
consume timestamp slots.

Consequently, neither `timestamp == instruction index` nor `start timestamp +
number of logged accesses` is valid. The transcript must delimit each instruction's
logical clock interval independently from its memory events.

It must also log reads, not only writes. A timed read changes a block's last-access
timestamp and therefore affects offline memory-checker and final touched-memory
rows even when the value is unchanged.

## Minimal preflight output

Preflight is an in-process phase, not a portable artifact format. The program,
configuration, initial state, and final `VmState` already exist at the call site; we
should not serialize their identities into every segment. There is no schema
version, program digest, VM-config digest, header object, or generic observation
arena.

The baseline hot path appends only these arrays:

```text
program_log: [(pc, start_timestamp), ... , (final_pc, final_timestamp)]

memory_log:  [(timestamp, address_space_and_kind, pointer, post_value), ...]

initial_write_log:
    [(address_space, pointer, initial_value), ...]
```

`program_log` uses two `u32`s per fetched instruction and one final sentinel. The
vector index is execution order. The last entry is never counted or routed. On
normal termination, the TERMINATE instruction is the last fetched entry before
the sentinel and has an empty clock interval. On suspension, no synthetic
instruction is fetched: the sentinel contains the resume PC and final timestamp.
Thus the invariant is always `program_log.len() == fetched_instructions + 1`, but
only a terminated segment has a TERMINATE entry.

The current VM guarantees that every non-TERMINATE instruction advances the memory
clock. Therefore the timed memory events belonging to instruction `i` are exactly
those with timestamps in:

```text
[program_log[i].start_timestamp, program_log[i + 1].start_timestamp)
```

Postflight indexing finds those ranges with one linear merge; preflight does not
write an access cursor per instruction. If a future opcode legitimately has a
zero-clock transition, this invariant must be revisited explicitly rather than
silently adding metadata for every current instruction.

Every proof-visible timed read and write appends one `memory_log` event.
`pointer` is aligned and expressed in the address space's native field-cell
addressing. `post_value` is four canonical `u32` cells: the observed value for a
read and the new value for a write. Packing is selected by the configured address
space layout: each cell is range-checked and encoded canonically, including field
elements in field-valued spaces. Invalid or non-canonical values fail closed.
`address_space_and_kind` packs the read/write bit with the address-space ID. The
executor log is not generic over the proof field.

The schema relies on a current VM-wide invariant: one memory-bus interaction is
exactly `BLOCK_FE_WIDTH == 4` field cells in every address space. Artifact
construction and Rust/C layout checks must reject a different width; silently
truncating or widening an event is forbidden.

Previous timestamps are derived by grouping memory events by
`(address_space, pointer)`. Previous values are also derived except when the first
timed event for a block is a write. Only that case appends one entry to
`initial_write_log`. The predecessor timestamp for the first event touching any
block in a segment is the segment baseline, currently zero; the first executable
timestamp is one. This convention is segment-local and does not depend on a prior
segment's last timestamp.

`initial_write_log` is the compact finalized result, not a mandate to do a random
lookup for every write. M1 must benchmark two hot-path strategies: a segment-local
sparse seen structure that emits the entry immediately, and sequential first-write
candidates that are filtered during cold finalization against the first timed event
of either kind for each block. Merely deduplicating writes is incorrect when a read
precedes the first write. The implementation should choose the lower end-to-end
cost and report both bytes written and lookup overhead. Either way, the returned
log contains exactly one entry only when a block's first timed access is a write.
This is less data than logging a previous value on every write and much less than
copying or logging whole Merkle leaves.

The initial value of every changed block can then be recovered from either its
first read or its `initial_write_log` entry. Untouched sibling blocks retain their
value in the complete final memory image. Thus system tracegen can construct a
separate read-only initial view as a complete logical view over the existing sparse
final-memory representation plus a sparse overlay of first-access values. This
does not require dense materialization of an address space. It must not apply that
overlay to the mutable final `VmState`, because that state is the next segment's
initial state. Initial Merkle
authentication state is owned by the proving/continuation coordinator, which may
instead retain the pre-mutation tree. It is not part of the preflight log.

### Untimed observations

A memory peek is not a memory-bus event: it neither increments the clock nor marks
the block touched. It therefore does not belong in `memory_log`.

Replay resolves a peek from the immutable version index as the value immediately
after the timed-event prefix already consumed by that instruction. This logical
point is `(instruction ordinal, local access cursor, current timestamp)`, not just
the timestamp. That distinction makes all of these cases unambiguous:

- a peek before the instruction's first timed access reads the version visible at
  instruction entry;
- a peek after a logged write in the same instruction reads the new version;
- several peeks at the same timestamp are ordered by replay even though none ticks;
- a block observed only by peeks reads directly from the reconstructed initial
  view because no timed event changed it.

The version index is built after preflight from `memory_log`,
`initial_write_log`, and final memory. It may materialize frequently used peek
results keyed by `(instruction ordinal, local peek ordinal)` if profiling shows
that repeated version lookups are expensive. That is derived tracegen data, not
another preflight write.

A proof-visible memory read must still be a timed `Read`; a peek-only value is
uncommitted execution advice. This is why omitting peeks from `memory_log` does not
lose a memory-bus interaction.

Memory peeks and non-memory host observations are different. Hint, random, and
deferral logic runs once during serial preflight. Whenever its proof-visible output
is materialized through ordinary memory writes, those writes are already sufficient
for replay. If an existing instruction consumes a non-deterministic value that is
not materialized in memory, preflight must append that value to an observation log
and replay must consume it rather than repeat the side effect.

That observation log should contain only concrete observation kinds required by
current instructions, with fixed layouts where possible. It should not reserve a
generic byte arena, per-instruction observation cursor, or extensible schema before
a real consumer needs one. Streams, RNG state, deferral caches, printing, and other
host state remain in `VmState` and are never re-executed by tracegen.

Accordingly, M1's RISC-V baseline has exactly the three logs above. A fourth,
instruction-specific observation array is added only if the M0 audit identifies a
current proof-visible input that cannot be recovered from static instructions,
immutable execution inputs, or logged memory. Final stream, RNG, and deferral state
can validate endpoints but cannot reconstruct an ordered history. The concrete
observation layout and measured write cost must be part of that change; it is not
an always-present transcript component.

Timestamps restart at one per continuation segment. The executor returns the three
arrays, final PC/timestamp/exit status, and the normal final `VmState`. Finalization
also extends `AddressMap::touched_pages` from timed accesses before that memory is
used by another segment or sparse host-to-device transfer.

## RVR executor design

### A real preflight execution kind

Add `RvrExecutionKind::Preflight` and a corresponding generated runtime/header.
The dormant `EmitMode::ValueTrace` is a useful seam, but it is not a preflight
implementation today:

- no public RVR execution kind selects it;
- no generated tracer state or host runtime backs it;
- extension page-tracing paths do not support it;
- register-zero shortcuts return before some value hooks;
- its scalar host loads do not express OpenVM's aligned block-access schedule;
- it cannot represent logical clock holes.

The new mode should reuse the existing per-instruction `trace_pc`/value-tracing
insertion points where their semantics match, but its contract is new.

### Runtime state

The generated function receives only the normal `RvState` through the block ABI.
A preflight runtime reachable through `RvState::mode_state` owns raw
pointers/capacities for the append buffers, the current timestamp, first-write
detection state, and cold error/grow callbacks. These cursors must not become extra
`preserve_none` block arguments: on x86-64 each one would evict a hot guest register
from the limited argument-register budget. M1 records the generated hot-register
count and rejects an ABI change that reduces it. There is no reason to force the
implementation into Rust collection types.

The likely fast implementation emits the append operations inline in generated C,
as pure and metered execution already emit mode-specific bookkeeping inline. Rust
owns allocation and consumes the completed buffers; it should not receive an FFI
callback per instruction or memory access. We can prototype more of this in Rust
if convenient, but the choice is made by generated-code size, native instruction
count, and end-to-end preflight benchmarks.

A plausible wire layout is deliberately plain C:

```c
typedef struct { uint32_t pc, timestamp; } ProgramEvent;
typedef struct {
    uint32_t timestamp, address_space_and_kind, pointer;
    uint32_t value[BLOCK_FE_WIDTH];
} MemoryEvent;
typedef struct {
    uint32_t address_space, pointer;
    uint32_t initial_value[BLOCK_FE_WIDTH];
} InitialWrite;
```

Buffers are reusable uninitialized chunks. The fast path is a capacity check plus
sequential stores; a cold grow callback may allocate another chunk. Overflow,
pointer overflow, and allocation failure stop execution with a typed error. Silent
truncation is forbidden.

Beginning an instruction appends `(pc, timestamp)`. Only a successfully completed
terminate or suspend run appends the final sentinel, using the termination PC or
suspension resume PC respectively. Any
guest trap, bounds failure, callback failure, timestamp overflow, allocation
failure, or other infrastructure error returns `Err` and discards the entire
transcript. A faulting instruction is never exposed as a retired step. Host helpers
must return typed errors rather than abort the process.

The executor does not promise transactional rollback of an in-place `&mut VmState`:
memory and host state may already have changed when an error is discovered. The new
owned API should therefore consume the working `VmState` and return it only on
success. If compatibility temporarily requires an `&mut` entry point, its contract
must mark the state poisoned and unusable after `Err`; callers may not retry or
continue from it.

### Log logical OpenVM operations

Instrumentation belongs in each opcode's RVR emission, before native optimization,
not around the loads and stores that survive C compilation. The emitter and replay
code must perform the same logical accesses, and tests must compare that schedule
with the AIR adapter. This does not require a new trait or access-spec registry, and
execution code must not import AIR code. Generated helpers model:

- register reads are timed even when the register is `x0` where current semantics
  require a memory-bus read;
- disabled and `x0` writes advance the logical clock without a memory event;
- byte and unaligned accesses expand to one or two aligned
  `BLOCK_FE_WIDTH` events in the same order as the AIR;
- a skipped second block still consumes its reserved clock slot;
- memory peeks call a typed untimed helper without appending an observation;
- phantom operations advance the prescribed clock but do not replay their host
  effect.

This makes the transcript a trace of OpenVM execution, not of whichever operations
the host compiler happened to retain.

### No mutation may bypass the logger

In preflight mode, every proof-visible memory path uses one common API, including
extension callbacks. Current hint/public-value callbacks and deferral code can
write raw address-space pointers behind generated C; those paths must instead call
logger-aware host functions that perform the mutation and append the corresponding
logical events. Direct proof-memory pointers are not exposed to preflight
extensions.

Artifact construction must know whether every registered RVR extension supplies
preflight-safe emission and callbacks. This can be a flag on existing extension
registration; it does not require a new extension trait. Preflight exposes only
typed memory, clock, and fallible host-call operations. Opaque C receiving raw
proof-memory pointers is rejected, and artifact construction fails if any extension
has not opted into the safe path.

Each append reserves its storage before the corresponding mutation. Variable-size
host operations may perform untimed sizing first and reserve their entire output
footprint before invoking a side effect. If any later operation fails, the full run
is discarded as described above; partial cursors or mutated working memory are
never returned as an authoritative result.

All timestamp advances are checked against the `timestamp_max_bits` bound committed
by the VM configuration. The executor must suspend at an already-planned segment
boundary or fail before an interval endpoint leaves that domain. RVR basic blocks
are already capped at 1000 instructions, but instruction count is not a timestamp
bound: metering can be timestamp-blind, and one variable-length instruction may
consume many clock slots. Segmentation must therefore charge the exact logical
timestamp cost, or preflight must prove a conservative per-instruction bound before
executing the instruction. A single instruction that cannot fit fails before any
of its effects are made authoritative.

### Segmentation

The first executor milestone runs to termination, subject to the checked timestamp
domain. It does not claim a bounded ABI.

The next executor milestone composes preflight logging with RVR's block-aligned
countdown state (`retired`, target boundary, and suspension status). It does not
pretend that an interpreter-derived instruction budget can stop generated RVR code
in the middle of a compiled block. The proving path first chooses a reachable block
boundary using RVR metering over the same compiled CFG; preflight must stop at and
validate that exact `(pc, retired)` boundary. A block that cannot fit within the
timestamp domain is rejected until finer CFG splitting is implemented. Silently
overshooting a continuation boundary is unsound.

## Postflight indexing

After serial execution, freeze the chunks behind an immutable shared handle. A
backend-independent indexing phase derives:

1. a PC histogram for the program trace, excluding the sentinel;
2. the instruction lists needed by registered trace generators, by looking up each
   fetched PC in the immutable program and considering opcode plus operands;
3. stable predecessor links for memory events grouped by
   `(address_space, pointer)` and ordered by `(timestamp, event ordinal)`;
4. current value/version at each event, seeded by the first read or sparse
   `initial_write_log` entry;
5. the last event and final value for every touched block;
6. row counts and prefix sums for variable-row traces.

These indexes are caches. They contain no new execution facts and can be rebuilt
or checked independently. Initially they may be built on CPU for simplicity. The
GPU path may upload the raw transcript once and perform grouping, stable sorting,
prefix scans, and producer bucketing on device when that wins in benchmarks.

Chronological logging followed by opcode bucketing creates gathers. We should
measure indirect step/access views, structure-of-arrays materialization, and value
versioning before choosing a device layout. A device-local permutation of generic
step and access indices is allowed. Reintroducing per-chip witness records is not.

## Read-only replay

For one program-log entry, tracegen needs only the static instruction, start/end
timestamp, its derived memory-event range, and the shared version index. A small
local cursor may expose operations such as:

```rust
read(address_space, pointer, timestamp) -> MemoryBlock
write(address_space, pointer, timestamp, expected_new_value) -> MemoryBlock
peek(address_space, pointer, logical_point) -> MemoryBlock
advance_timestamp(slots)
finish(expected_next_pc)
```

These are memory-bus operations, not heap-specific operations. `address_space`
selects registers (`RV64_REGISTER_AS`), main memory, public values, deferral memory,
or another configured space. `MemoryBlock` means the fixed `BLOCK_FE_WIDTH` cells
carried by one memory-bus interaction. Opcode implementations use semantic helpers
above this cursor—for example `read_register`, `write_register`, or
`read_main_memory`—so register and heap behavior remain explicit while sharing one
minimal log representation.

Each operation checks the expected access kind, address space, aligned pointer,
timestamp, and value. A write returns the indexed previous value and checks the
recorded new value. `finish` requires exact exhaustion of the access slice plus the
recorded end timestamp.

Deterministic instruction outputs are always independently computed by producer
semantics and compared with the transcript. GPU replay accumulates every cursor,
schedule, address, and value mismatch into a device error flag which is reduced and
checked by the host in release builds before a trace is accepted. More expensive
full legacy-trace comparisons remain differential-test-only.

This operational check is non-negotiable. Without it, removing records merely
moves silent execution/AIR drift into opaque replay kernels.

Replay is side-effect-free. It does not mutate guest memory, streams, RNG, metrics,
or host state. Independent invocations may therefore replay in parallel even when
their original executions had memory dependencies.

## Trace-generator ownership

Do not design a new prover trait as part of the executor milestone. Initially keep
the existing chip/filler registration and change one trace generator at a time to
consume indexed log views instead of records.

The only architectural rules needed now are:

- execution maps an instruction to RVR semantics without consulting a chip;
- postflight routing looks at the static instruction, including operands when an
  opcode such as PHANTOM is shared;
- exactly one trace generator emits each non-TERMINATE instruction's execution-bus
  transition and program lookup;
- any number of additional traces may consume the same instruction but must not
  duplicate those interactions;
- TERMINATE remains system-owned.

This supports many opcodes sharing a trace and one instruction feeding several
traces without putting trace IDs in preflight. Keccak and SHA-256 can fan out to
their operation and permutation/block-hasher traces from the same log entry.

Dynamic row expansion also remains tracegen work: compute row counts after
preflight, prefix-scan them, allocate output, and fill it. Trace generators emit
their lookup requests; range-check, bitwise, Poseidon, and other dependent traces
run after the required reductions. We should introduce a new common API only after
the first few RISC-V GPU ports show what they actually share.

## System trace generation

The same transcript covers the system AIRs:

- **Program:** count fetched step PCs against the immutable program, including the
  terminating PC exactly once when termination occurs.
- **Connector:** use the input execution state, returned final state, and exit
  reason already owned by the caller. The connector remains the sole owner of the
  TERMINATE program lookup and its timestamp range-check requests; RISC-V replay
  must not duplicate them.
- **Instruction memory columns:** the trace that owns an instruction uses indexed
  predecessor timestamp/value plus each event's current timestamp and post-value.
  These columns live in adapter traces; there is no standalone offline-memory
  trace.
- **Memory boundary:** reconstruct initial touched leaves from final memory plus the
  sparse first-access overlay, then use each touched block's final value and last
  timed-access timestamp. A read-only touched block is included; a peek-only block
  is not.
- **Persistent memory and Merkle:** derive changed/touched leaves from the same
  index and use coordinator-owned initial Merkle state or the reconstructed initial
  view, then generate boundary, Merkle, and Poseidon work in dependency order.
- **Public values:** after the final segment, read the final public-value address
  space and produce its Merkle proof against the final segment's completed Merkle
  top tree; there is no public-value system trace.
- **Phantom:** its primary opcode producer uses the fetched instruction and start
  timestamp; it does not repeat the host side effect.

Initial memory transfer/root construction remains a separate step before mutation.
Continuations create one self-contained transcript per segment, carry final mutable
VM state forward, and never retain event references across segments.

## Migration plan

### M0: lock down semantics

- Add reference tests that extract PC, timestamp, timed access, touched-memory,
  touched-page metadata, and final-state facts from current interpreter preflight.
- Cover register `x0`, disabled writes, immediates, crossing and non-crossing
  unaligned loads/stores, all address spaces, peeks, phantom, hintstore, public
  values, RNG, deferral, termination, traps, timestamp limits,
  peek-before/after-write, and continuation boundaries.
- Assert for every currently registered non-TERMINATE instruction that the end
  timestamp is greater than the start timestamp; this justifies deriving access
  ranges without logging cursors.
- Define the three fixed-width C buffer layouts and assert their Rust-side size and
  alignment. This is an internal build interface, not a versioned file format.
- Assert the segment baseline predecessor timestamp, per-address-space canonical
  packing, and the global `BLOCK_FE_WIDTH == 4` assumption at the Rust/C boundary.

### M1: RVR preflight executor only

- Add `RvrExecutionKind::Preflight`, generated logical-access helpers, the append
  runtime, and transcript finalization.
- Return the three logs, final PC/timestamp/exit, and final VM state. Any
  initial-memory/Merkle state retained by the proving coordinator stays outside the
  executor API.
- Leave the production prover and `RecordArena` unchanged.
- Initially enable only configurations whose complete RVR extension set opts into
  preflight-safe emission; reject all other artifact builds. Logger-aware
  IO/public-value, hint, and deferral callbacks are executor work and must land
  before those extensions are enabled.
- Differentially compare RVR preflight with interpreter preflight for PC sequence,
  instruction clock intervals, access sequence, final memory/registers, touched
  blocks/pages, streams/RNG/deferral state, exit, and program frequencies. Error
  cases must return no transcript or final-state result.
- Benchmark the executor immediately against interpreter preflight as the M1 gate,
  and also against pure and metered RVR. Measure bytes appended and native
  instructions per guest instruction. Compare inline generated C with any
  Rust-assisted prototype before committing to the implementation.
- Compare the segment-local sparse-seen and append-then-deduplicate strategies, and
  verify that preflight adds no block-ABI arguments or loss of hot guest registers.

This is the first implementation step and the first independently useful result.

#### Initial executor checkpoint (2026-07-23)

The first fail-closed executor slice supports RV64I/M register operations,
control flow, loads/stores, and termination. It rejects phantoms, hint/IO
callbacks, and extension memory wrappers until their complete schedules are
logged. Exact native tests cover x0 reads and disabled writes, non-crossing
second-access gaps, crossing loads/stores, first-write filtering, capacity
failure, and the final sentinel.

On an x86-64 host, a seven-run differential benchmark over a hand-built
2,003-instruction RV64I loop produced identical final PC/timestamp, per-PC
frequencies, and touched-memory values/last timestamps. Median execution time
was 2.600 ms for interpreter preflight and 0.411 ms for RVR preflight (6.31x).
The three logs occupied 128,192 bytes, or 64.0 bytes per guest instruction.
This is an implementation checkpoint, not the M1 gate: the pinned suite must
also cover memory-heavy programs and logger-aware host callbacks before M2.

### M2: RISC-V GPU feasibility slice

Before designing the rest of tracegen, test the central bet on the most important
backend: GPU tracegen for the existing RISC-V AIRs.

- Upload the three compact logs and build only the indexes needed by RISC-V.
- Port RISC-V trace generation directly to read-only replay, starting with simple
  fixed-row arithmetic and then registers, branches, loads/stores, unaligned
  accesses, multiplication/division, and Phantom.
- Include the offline-memory columns inside those RISC-V adapter traces.
- Compare every generated matrix and lookup request with legacy GPU tracegen on the
  same executions.
- Measure log bytes per instruction, host append throughput, transfer time,
  indexing/sort time, gather efficiency, device bandwidth, kernel time, and total
  RISC-V tracegen time.

This is a decision gate. If chronological logs make GPU replay uncompetitive, fix
the log/index layout here before porting system AIRs or other extensions. Do not
hide the result behind a generic tracegen framework or a CPU-first implementation.

The gate passes only if all of the following hold on a fixed representative RISC-V
suite chosen before tuning:

- generated RISC-V matrices match the legacy GPU path exactly wherever row order
  is defined, and canonicalized lookup-request multisets are equal otherwise;
- the new path contains no production `RecordArena` or chip-shaped compatibility
  records;
- peak memory is no larger than the legacy path's peak over the same phase and
  inputs, with identical trace-output allocations either included on both sides or
  excluded from both;
- executor + upload + indexing + RISC-V kernel time has no more than 10% regression
  on any representative workload and is no slower in geometric mean than legacy
  interpreter preflight + RISC-V GPU tracegen.

Failure means changing the log/index layout and repeating M2, not proceeding to
system AIRs. The raw measurements remain in the repository even if the gate fails.
Before tuning, commit the benchmark manifest containing the programs and input
sizes, GPU and host model, compiler flags and versions, warmup count, repetition
count, and reported statistic. The 10% per-workload threshold is evaluated with a
documented noise interval rather than a single run.

#### M2 GPU benchmark manifest

The first performance gate uses a fixed `ADDI x2, x2, -1; BNE x2, x0, -4`
loop with `x2 = 2^18` in initial register memory, followed by TERMINATE. It
therefore produces exactly `2^18` ADDI rows with interleaved branch accesses and
no trace padding. The benchmark is a release build on the dedicated GPU host,
with 3 warmups and 11 measured samples. Each CUDA sample contains 10 launches;
legacy and replay order alternates between samples. Results report median,
p10, and p90.

Serial RVR and interpreter preflight, synchronized segment H2D/indexing, and
legacy-record H2D are timed separately. Initial-state construction, VM keygen,
metering, RVR artifact compilation, and trace allocation are setup costs and are
excluded from both timed paths. Direct legacy and replay kernels reuse
pre-uploaded inputs and preallocated equal-sized output matrices, and use CUDA
events on the same stream. Static-program upload is reported separately because
it is reusable across segments. The run records CPU, GPU, driver, rustc, and nvcc
versions and checks that no competing GPU process is active.

The first two runs on 2026-07-23 used an AMD EPYC Milan host (16 physical
cores), NVIDIA L40S (46,068 MiB), driver 610.43.02, rustc 1.93.0 with LLVM
21.1.8, and CUDA 13.3 (`nvcc` 13.3.33). The exact command was:

```text
cargo test --release -p openvm-riscv-circuit --features cuda,rvr \
  benchmark_cuda_addi_replay_vs_legacy -- \
  --ignored --nocapture --test-threads=1
```

This uses the workspace release profile (`opt-level = 3`, thin LTO, 16
codegen units, line-table debug information) and the repository's default CUDA
build flags. No other GPU compute process was active. The two median results
were stable:

| Stage | Run 1 | Run 2 |
| --- | ---: | ---: |
| Cold host postflight index | 54,006.7 us | 53,549.4 us |
| Replay transcript/index H2D | 2,046.5 us | 2,051.5 us |
| Legacy ADDI-record H2D | 503.4 us | 507.8 us |
| Replay ADDI kernel | 56.94 us | 57.14 us |
| Legacy ADDI kernel | 29.39 us | 29.58 us |
| Replay / legacy kernel | 1.938x | 1.931x |

The transcript is 33,554,448 bytes, the derived replay index is 8,388,616
bytes, and the equivalent ADDI-only legacy records are 11,534,336 bytes. The
comparison deliberately charges the replay path for the shared ADDI+BNE
transcript and index while the legacy byte count contains only ADDI records.
The conservative median total is therefore 55.7 ms for replay versus 0.54 ms
for the isolated legacy ADDI path. This slice is far outside acceptable
overhead: the first required optimization is the postflight/index
representation, and the replay kernel's extra validation and gather cost must
also be reduced before porting more AIRs. It does not by itself decide the
formal end-to-end M2 gate because it excludes both serial execution paths and
the legacy BNE record upload/kernel while charging the complete shared
ADDI+BNE transcript and index to ADDI.

As a CPU-oracle optimization, packing the block key into a `u64`, using the
repository's deterministic `FxHashMap`, updating each predecessor entry with a
single lookup, and fusing timestamp validation into the step scan reduced the
cold index median from 53.5 ms to 24.0 ms. A measured attempt to replace the
small opcode `BTreeMap`s with hash maps regressed it to 35.0 ms and was
discarded.

The next checkpoint moved both indexes to the GPU. Memory indexing packs a
32-bit block label and a 32-bit source ordinal into one unique `u64` radix-sort
key. First-write seeds occupy the ordinal prefix and therefore sort before
chronological memory events for the same block without relying on sort
stability. A second, stable radix sort derives replay steps partitioned by
opcode; only the small per-opcode range table returns to the host. The static
GPU artifact binds the program and `MemoryConfig`, rejects a block-label domain
wider than 32 bits, and the kernels reject out-of-range or unaligned addresses.
The production GPU path no longer calls the CPU replay-index builder or uploads
its 8.4 MiB output. That builder remains only as a differential-test oracle.

On the same fixed workload, two compact-key runs measured 1,623.2 and 1,752.4 us
median for raw-transcript H2D plus GPU memory indexing, 134.3 and 136.5 us for
GPU program indexing plus range-table D2H, and 56.7 and 56.8 us for the replay
ADDI kernel. Their conservative isolated totals are 1,814.3 and 1,945.7 us,
down from 25,857.3 us after the CPU-oracle optimization and 55.7 ms at the
initial checkpoint. They are 3.41x and 3.64x the corresponding isolated legacy
ADDI record-upload-plus-kernel slices, with the same deliberate asymmetry:
replay is charged for the complete ADDI+BNE transcript while legacy includes
only ADDI records. The retained GPU-derived state is 8,388,616 bytes. Replacing separate
64-bit block keys and 32-bit ordinal values with the unique packed key reduced
peak requested live-buffer payload from 42,240,511 to 29,657,599 bytes above
the raw transcript and static program. CUB's
double-buffer overload had previously produced the same queried workspace and
latency as ordinary pair sorting and was discarded. The compact-key result is a
material improvement. Staging predecessor allocation after the radix sort then
reduced that requested payload again to 25,463,295 bytes: the input keys and CUB
workspace are freed on the same stream before the 4,194,304-byte predecessor
buffer is allocated. This is a live requested-byte calculation, not a physical
device-memory measurement: allocator page rounding, the four-byte shared error
word, and the raw-log/static-program baseline are excluded. The formal M2 gate
must additionally measure the allocator high-water delta over an identical
phase. Two runs measured 1,781.2 and 1,610.5 us for transcript H2D plus
memory indexing and conservative totals of 1,975.2 and 1,799.4 us, so the split
adds no measured latency regression. The result still fails the isolated
slice's peak-memory target. Porting BEQ/BNE next is necessary to remove the
remaining legacy RISC-V records from this exact workload and make the complete
RISC-V phase comparison symmetric before deciding whether the index must change
again.

#### Initial GPU correctness checkpoint (2026-07-23)

The first direct replay kernel covers ADDI without constructing a compatibility
record. The three logs and static program are uploaded once. Cold postflight
builds one packed predecessor reference per memory event and one stable,
opcode-partitioned step buffer shared by every kernel. Replay validation uses a
single device error word that is checked once after the kernel batch, rather than
synchronizing after every AIR.

The CUDA differential test executes a real RVR loop with a repeated ADDI PC and
interleaved BNE memory events. The replay ADDI matrix matches both the legacy GPU
kernel and CPU filler exactly, its raw range-check histogram matches legacy
exactly, and the resulting AIR proof passes. Corrupting a logged ADDI result is
rejected by device validation. ADDI with `rd = x0` is deliberately rejected:
the current immediate-ALU AIR always emits a destination write and must gain the
same conditional-write shape as load/JALR before that schedule can be replayed.

This establishes correctness and log sufficiency for one fixed-row RISC-V AIR,
and removes CPU postflight from the production GPU path, but does not yet
exercise the complete M2 performance gate. The next checkpoint is direct
BEQ/BNE replay, followed by a full-path executor plus RISC-V comparison before
widening to the other RISC-V adapters.

#### Branch replay and full-slice checkpoint (2026-07-23)

The second direct replay kernel covers BEQ and BNE. It reconstructs both timed
register reads through the shared predecessor index, validates the exact clock
schedule and branch target, and writes the adapter and core columns directly.
The production path does not construct an `Rv64BranchAdapterRecord` or use a
`RecordArena`. A shared CUDA helper now implements only the mechanical operation
of resolving a four-cell predecessor value; instruction-specific schedules stay
in their concrete kernels.

The differential test includes a taken BEQ and an interleaved BEQ/BNE loop.
ADDI matrices match both legacy GPU and CPU fillers exactly. Branch matrices
match after canonicalizing their row order by timestamp because replay groups
opcodes while legacy records preserve execution order. Their combined
range-check histogram matches legacy, and a proof containing both AIRs passes.
Separate corruptions of ADDI, branch outcome, and predecessor consistency fail
device validation.
This test also fixed an important replay rule: when a block's first timed access
is a read, the event's logged value is the segment's initial touched value. It is
not required to be zero. A first-write predecessor instead comes from the sparse
initial-write log.

The same fixed workload was then measured symmetrically: both ADDI and BNE record
upload/kernels are charged to legacy, both replay kernels are charged to the new
path, and both serial preflight executors are included. The median result was:

| Stage | Replay | Legacy |
| --- | ---: | ---: |
| Serial preflight | 28,095.7 us | 52,792.0 us |
| Program GPU index | 133.9 us | N/A |
| Transcript H2D + memory GPU index | 1,634.8 us | N/A |
| Record H2D | N/A | 1,006.6 us |
| ADDI + BNE kernels | 125.1 us | 61.7 us |
| GPU phase total | 1,893.8 us | 1,068.3 us |
| Summed slice total | 29,989.5 us | 53,860.4 us |

RVR preflight is 1.88x faster than interpreter preflight on this workload. The
read-only replay GPU phase is 1.77x slower and its two kernels are 2.03x slower,
but the sum of the independently measured stage medians is 1.80x faster. The
transcript is 33,554,448 bytes, the steady derived index is 8,388,616 bytes, and
the two legacy record arrays total 24,117,248 bytes. This is a modeled RISC-V
slice total, not an observed contiguous pipeline sample: timed executor outputs
are discarded, and the GPU stages use equivalent inputs prepared outside those
samples. It also excludes system AIRs, continuation/Merkle work, proof
generation, and reusable static-program upload. The result passes the time
criterion for this one fixed feasibility slice without hiding the GPU cost.

They do not complete the formal M2 gate. The suite still lacks the other RISC-V
access shapes, and the 25,463,295-byte index figure is requested live-buffer
payload rather than an allocator high-water measurement over an identical phase.
The next checkpoint is therefore the remaining fixed-row RV64I arithmetic and
register adapters, followed by loads/stores and the actual memory-peak comparison.

The first widening step adds direct ADDIW replay. It authenticates the full
64-bit source and destination events while the core re-executes only the low
32-bit addition, then requires the logged upper limbs to equal the result's sign
extension. CPU, legacy CUDA, and replay matrices and range histograms match for
negative, positive, low-limb-carry, and in-place-destination cases; a corrupted
sign-extension limb is rejected. Writes to x0 remain fail-closed because the
transpiler canonicalizes those instructions to NOP and the AIR has no disabled
write row.

SLTI and SLTIU are the next fixed-row slice. Their shared replay kernel consumes
two explicit opcode ranges, validates the comparison result against the logged
write, and fills the existing immediate adapter and comparison core directly.
An interleaved-opcode test matches CPU and legacy CUDA rows after timestamp
canonicalization, matches the complete range histogram including the
equal-operands/no-difference path, rejects a corrupted result, and proves the
AIR. No comparison record is constructed by production replay.

SLLI and SRLI follow the same direct shape. Replay decodes the logged source,
re-executes the shift on the GPU, checks the logged destination, resolves both
memory predecessors, and fills the existing immediate adapter and shift core
without a compatibility record. The differential proof test starts from a
nonzero register in the untimed initial memory image and covers shifts 0, 1, 15,
16, 31, 32, and 63 across interleaved opcode groups. CPU, legacy CUDA, and replay
rows and range histograms match, while corrupting a nonzero cross-limb result is
rejected. This also exercises first-touch reads from initial memory without
adding a setup instruction or timed observation to the transcript.

SRAI then validates the sign-sensitive variant independently. Its direct kernel
preserves the core's sign decomposition and extra range request while checking
the logged destination against a GPU arithmetic shift before emitting any
lookup counts. The test covers both signs, x0 reads, an in-place destination,
and shifts 0, 1, 15, 16, 31, 32, 47, 48, and 63. Direct replay and legacy CUDA
match the CPU matrix and complete range histogram exactly, the replay trace
proves, and a corrupted sign-filled cross-limb result is rejected.

SLLIW and SRLIW extend replay to the word adapter. The kernel authenticates all
64 source bits, executes the shift over the low 32 bits, and requires both upper
destination limbs to equal the low-word result's sign extension. An interleaved
boundary test covers shifts 0, 1, 15, 16, and 31 and both positive and negative
word results. Replay and legacy CUDA match the CPU rows and range histogram, the
replay trace proves, and corrupting an upper sign-extension limb is rejected.

SRAIW completes the immediate word-shift family. It authenticates the full
source register, arithmetic-shifts its low signed word, and verifies the full
sign-extended 64-bit write. The replay test covers both sign boundaries, x0,
an in-place destination, shifts 0, 1, 15, 16, and 31, and non-power-of-two row
padding. CPU, legacy CUDA, and replay traces and range histograms match; the
replay trace proves and rejects a corrupted upper sign-extension limb.

XORI, ORI, and ANDI add the byte-limb and bitwise-lookup path. Replay validates
the canonical signed 12-bit immediate encoding, converts the four logged u16
cells to eight little-endian bytes, checks the computed full-register result,
and emits both timestamp range and bitwise lookup requests directly. A grouped
test covers all three opcodes with immediates -2048, -1, 0, and 2047, plus x0,
an in-place destination, and padding. CPU, legacy CUDA, and replay rows and both
lookup histograms match; the replay trace proves and rejects a corrupt result.

### M3: complete the GPU proving path

Once RISC-V replay is viable:

- generate Program, Connector, PersistentBoundary, MemoryMerkle, and system-origin
  Poseidon requests from the same logs;
- merge all RISC-V and system requests before generating the global range-check,
  bitwise, and other periphery traces;
- reconstruct initial touched leaves from final memory plus the sparse
  first-access overlay in a separate complete logical view of the sparse final image,
  or reuse coordinator-owned initial Merkle state; never mutate the carried final
  state while reconstructing the initial view;
- validate continuation boundaries, final public-value extraction, and Merkle
  proofs;
- run full proof equivalence against the legacy path.

### M4: other GPU extensions

Port extensions one at a time after the RISC-V and system path works:

- Keccak or SHA-256 first to exercise one instruction feeding multiple traces;
- bigint, algebra, and ECC;
- hintstore, deferral, and host-callback-heavy extensions;
- remaining extensions and their lookup/periphery dependencies.

Each extension's RVR execution path must support safe preflight logging before its
tracegen is enabled. Logger-aware callbacks are required for every proof-visible
address space.

### M5: CPU tracegen if still required

GPU feasibility comes first. If CPU tracegen remains a supported backend, port it
after the log and replay model has been validated on GPU. It should consume the
same postflight indexes rather than introduce another record representation.

A test-only log-to-legacy-record adapter is acceptable as an oracle during
migration. It must not enter the production architecture.

### M6: cut over and delete records

- Switch proving to RVR preflight plus transcript replay.
- Remove `PreflightExecutor`, record arena sizing, per-chip record types used only
  for preflight, and interpreter preflight.
- Remove `RecordArena` only after repository-wide search shows no execution,
  system, CPU tracegen, GPU tracegen, or test consumer remains.
- Revisit whether metered execution can collapse to a single segmentation metric
  after preflight and proving use the same RVR block model.

## Required invariants

The implementation is acceptable only if all of these remain true:

1. The baseline three logs contain no chip, AIR, executor, trace-height, schema, digest, or
   record-layout identifier.
2. Every fetched PC is ordered, including termination; instruction order is not
   inferred from timestamps.
3. Every logical timed read and write produces exactly one canonical block event.
4. Clock-only slots advance the timestamp without producing fake memory events.
5. Peeks neither advance the timestamp nor mark memory touched.
6. No proof-visible address-space mutation bypasses the logger.
7. Replay consumes exactly the timestamp-derived instruction slice and reproduces its complete
   logical access and clock schedule.
8. Host side effects run once during serial execution and never during replay.
9. First reads, sparse initial-write entries, timed events, and the complete final memory image derive
   every previous value/timestamp, initial touched leaf, and final touched block.
10. A single step may feed multiple traces without duplicating execution
    metadata.
11. Segment-local clocks and endpoint states are explicit; initial Merkle state is
    coordinator-owned and no event reference crosses a continuation boundary.
12. Capacity exhaustion, ABI mismatch, unsupported raw callbacks, and malformed
    transcripts fail closed.
13. Exactly one trace emits each non-TERMINATE instruction's execution and program
    interactions; additional traces cannot duplicate them.
14. `initial_write_log` contains exactly one entry only when a block's first timed
    access is a write; returned final memory carries complete touched-page metadata.
15. A failed/trapped run returns no transcript or final-state result, and every
    timestamp endpoint stays inside the configured proof domain.
16. Peeks append nothing and create no memory-proof claim.
17. The first event for a block in every segment has predecessor timestamp zero,
    and no predecessor link crosses a segment boundary.
18. Every logged memory event is exactly four canonically packed field cells; a
    configuration with another memory-bus width is rejected.
19. Reconstructing an initial Merkle view never mutates the final `VmState` carried
    to the next segment, and final public values use the final segment's Merkle top
    tree.

## Performance criteria

The design aims to make host preflight an append-dominated workload, but it does
not assume it is memory-bandwidth optimal. We should track:

- native instructions and bytes appended per guest instruction;
- timed accesses and clock-only slots per guest instruction;
- executor throughput relative to plain RVR execution;
- transcript bytes before and after indexing/compression;
- seen-bitmap cost and bytes in `initial_write_log`;
- host-to-device transfer time;
- indexing/sort/scan time;
- replay gather efficiency and achieved device bandwidth;
- tracegen time per trace and end-to-end proving time.

Optimizations are accepted only when they preserve the semantic transcript and
improve end-to-end measurements. Per-chip record buffers, proof-specific metadata
on the serial hot path, and unchecked raw memory callbacks are outside the design
even if locally faster.

## Deliberate non-goals and open measurements

- This design does not solve arbitrary-address `JALR` or basic-block discovery.
  It uses the RVR CFG and its current block-boundary execution contract.
- It does not redesign the AIRs or decide whether an opcode deserves a
  super-instruction.
- It does not promise fully concurrent generation of dependency-related periphery
  traces.
- AoS versus SoA event layout, value interning, and CPU versus GPU indexing remain
  benchmark choices.
- Atomic LR/SC reservation state is absent from the current VM. A future atomic
  extension must give reservations an explicit transcript/state contract rather
  than infer them from ordinary memory accesses.

The stable boundary is intentionally small: `(pc, timestamp)`, canonical timed
memory accesses, rare initial-write values, and segment endpoints. Everything
chip-shaped is derived after execution.
