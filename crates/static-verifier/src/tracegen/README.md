# tracegen

Parallel witness generation for `StaticVerifierCircuit`. The circuit's
populate trace is recorded once as a dataflow IR, then interpreted in
parallel against a shared witness tape.

## Architecture

Three files, one invariant: **executing the IR in tape order reproduces
the halo2 backend's advice + range tapes byte-for-byte.**

### `ir_builder.rs` — `Halo2IRBuilder`

Implements the same `chip_traits` trait set as `Halo2Backend`, but each
method call records a `Halo2GraphNode` instead of assigning halo2 cells.
Every node writes a statically-known slice of the advice and range tapes.

- `max_bits` tracking and explicit `BBReduce` nodes for pre-op reduces.
- Constant caches (`Context::load_zero`, `BabyBearChip::const_cache`,
  `TranscriptChip`'s own baby-bear cache). Atomic ops (`BBDiv`, `ExtMul`,
  `ExtDiv`) expose internally-loaded constants as extra outputs, and
  whether each cell materializes is decided at build time and recorded
  per node in `NodeMeta::constant_skip_inds`.
- Transcript sponge/buffer state matches `TranscriptChip`.

The builder also records a `NodeMeta` per node (tape offsets, lengths,
skip indices, operand offsets) by replaying the op on a
`CalculateOffsetsTape` seeded with the current cache state.

### `opcode_impl.rs` — standalone opcode replay

`run_op<T: ReplayTape>` executes one `Halo2Opcode` against a tape
implementation. Two implementations share the same op logic:

- `CalculateOffsetsTape` — build-time. Records values + output offsets to
  derive `OpcodeMeta` (ctx/lookup lengths, output offsets, constant-skip
  indices); carries a constant cache seeded with the builder's warm set.
   - We use this to derive the node metadata that is necessary to keep this impl byte for byte
   - this means adding some extra tracking info for constants that may or may not be written onto the advice tape due to constant caching in the circuit vk construction (TODO: make this better if vk can be changed)  
- `WitnessTape` — runtime. Streams values into caller-provided buffers
  via raw pointer bumps. Stateless: `constant_skip_inds` precomputed at
  build time tells it exactly which `load_constant` calls write a cell.
   - this is optimized to run as fast as possible

Gate, range, and BabyBear primitives are provided methods on
`ReplayTape`, so one set of implementations serves both build-time
metadata derivation and runtime replay.

### `graph_executor.rs` — parallel interpreter

Lowering (`GraphProgram::lower`) flattens the IR into a flat
`Vec<GraphCoreInst>` over a single tape laid out as
`[advice | lookups | consts]`. 

The Executor is split into two components: `GraphProgram` and `GraphExecutorState`. The former contains the constants/instructions/metadata needed for the executor to run, that's not mutable, while the latter contains the mutable components. This is so that `GraphProgram` can be part of the `pk`. 

`GraphExecutor` borrows both, the former immutably and the latter mutably.

Execution runs in two phases:

1. Input population: `load_proof_wire` writes each `LoadWitness`'s cell.
2. `GraphExecutor::run` claims level-sorted compute instructions off a
   shared atomic cursor. Each worker spin-waits on parents' `AtomicU8`
   flags (Release/Acquire) before executing — no barriers. Meanwhile the
   calling thread walks the emission-order release schedule and streams
   newly-materialized tape ranges through an `on_delta` callback, with
   flushes batched between `MIN_FLUSH_CELLS` and `MAX_FLUSH_CELLS`.

`FusedColumnBuilder` is the intended `on_delta` sink: it writes tape
deltas directly into their halo2 advice-column rows (H2D on GPU,
`copy_from_slice` on host), so no host-side witness buffer is ever
materialized.

`GraphProver` wraps `GraphExecutor` with public-value extraction. It is
built once at keygen (see `StaticVerifierProvingKey::keygen`) and reused
across proofs of the same static shape.

## TODOs

Both require verifying-key changes and are therefore blocked on a vk bump.

1. **Refactor constant caching.** `TranscriptInst::transcript_load_reduced_constant`
   exists only because `TranscriptChip` keeps a baby-bear constant cache
   distinct from `BabyBearChip::const_cache`. Merging the two lets the
   trait method go away and drops the redundant constant cells that
   duplicate values already materialized by the outer chip.

2. **Graph-level optimizations.** The IR is a natural target for
   constant folding and common-subexpression elimination, but any node
   the optimizer removes changes the halo2 cell count and layout.
