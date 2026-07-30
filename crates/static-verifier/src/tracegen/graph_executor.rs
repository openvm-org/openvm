#![allow(rustdoc::private_intra_doc_links)]
//! Parallel executor for the graph IR recorded by [`Halo2IRBuilder`].
//!
//! Lowering flattens each graph node — using its
//! [`NodeMeta`](super::ir_builder::NodeMeta) — into a [`GraphCoreInst`] whose
//! operands are absolute tape offsets. The tape layout is
//! `[advice | lookups | consts]` (dedup'd fixed-column constants live in the
//! trailing region), so operand gathering is uniform.
//!
//! Execution has two phases:
//! 1. `load_proof_wire` streams proof witnesses into the tape via [`PopulateInputs`], replaying
//!    recorded `LoadWitness` instructions.
//! 2. [`GraphExecutor::run`] claims level-sorted compute instructions off a shared atomic cursor;
//!    each worker spin-waits on its parents' done flags (Release/Acquire) — no barriers.

use std::{
    collections::HashMap,
    sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering},
};

#[cfg(feature = "halo2-gpu")]
use halo2_base::halo2_proofs::cuda::utils::HALO2_GPU_CTX;
use halo2_base::halo2_proofs::{
    arithmetic::Field as _, halo2curves::bn256::Fr, plonk::AdviceColumns,
};
#[cfg(feature = "halo2-gpu")]
use openvm_cuda_common::d_buffer::DeviceBuffer;
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2Config as RootConfig,
    openvm_stark_backend::{
        p3_field::{BasedVectorSpace, PrimeField64},
        proof::Proof,
    },
    p3_baby_bear::BabyBear,
};
use serde::{Deserialize, Serialize};

use crate::{
    chip_traits::{ChipBase, PopulateInputs},
    circuit::StaticVerifierCircuit,
    field::baby_bear::{
        BabyBearExt4, BabyBearWire, ReducedBabyBearExt4Wire, ReducedBabyBearWire, BABYBEAR_MAX_BITS,
    },
    tracegen::{
        ir_builder::{GraphCell, Halo2IRBuilder, Halo2Opcode},
        opcode_impl::{interpret_op, UNMATERIALIZED},
    },
};

/// One lowered graph node, replayed via
/// [`interpret_op`](super::opcode_impl::interpret_op).
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
struct GraphCoreInst {
    opcode: Halo2Opcode,
    /// Slice of `operand_offsets` / `operand_bits`.
    args: (u32, u32),
    /// Slice of `const_inds` (the node's constant-skip indices).
    const_inds: (u32, u32),
    /// Absolute advice-tape offset the node writes at.
    ctx_offset: u32,
    ctx_len: u32,
    /// Absolute lookup-tape offset the node writes at.
    lookup_offset: u32,
    lookups_len: u32,
    /// Slice of `dep_inds` (parent instruction indices).
    dep_list: (u32, u32),
}

/// One compute writer in the emission-order release schedule. Ends are
/// precomputed at lowering so the release walk scans this array sequentially
/// instead of random-accessing the (level-sorted) `insts` per writer.
#[derive(Copy, Clone, Serialize, Deserialize)]
struct ReleaseEntry {
    /// Index into `insts`/`flags` (level-sorted position).
    inst: u32,
    /// Exclusive end of this writer's advice write range.
    advice_end: u32,
    /// Same, for the range-check tape.
    lookup_end: u32,
}

/// Shared mutable witness tape.
#[derive(Clone, Copy)]
struct TapePtr(*mut Fr);
#[allow(unsafe_code)]
unsafe impl Send for TapePtr {}
#[allow(unsafe_code)]
unsafe impl Sync for TapePtr {}

/// The immutable output of lowering a graph IR: everything needed to replay
/// the populate trace that is a pure function of the static circuit shape.
///
/// Serialized alongside the proving key; on decode a fresh
/// [`GraphExecutorState`] is paired with it (via [`GraphExecutor::new`]) to
/// replay the populate trace without re-recording it.
#[derive(Serialize, Deserialize, Clone)]
pub struct GraphProgram {
    advice_cells: usize,
    lookup_cells: usize,
    lookup_bits: usize,
    /// Dedup'd fixed-column constants; the trailing region of the tape.
    consts: Vec<Fr>,
    /// Flattened operand tape offsets across all instructions.
    operand_offsets: Vec<u32>,
    /// Bit bound per operand, parallel to `operand_offsets`.
    operand_bits: Vec<u16>,
    /// Flattened constant-skip indices across all instructions.
    const_inds: Vec<u32>,
    /// Compute instructions, level-sorted.
    insts: Vec<GraphCoreInst>,
    /// `LoadWitness` instructions, in emission order.
    input_insts: Vec<GraphCoreInst>,
    /// Compute writers in emission order (ascending write offsets on both
    /// tapes). The release walk in [`GraphExecutor::run`] polls writers in
    /// this order, folding the input-populated gap cells before each writer
    /// into the pending range.
    release_order: Vec<ReleaseEntry>,
    /// Flattened parent-instruction indices, sliced by `insts[i].dep_list`.
    dep_inds: Vec<u32>,
    /// Advice offsets of the circuit's public values, in instance order.
    /// Empty when lowered from a constraints-only trace.
    pv_offsets: Vec<usize>,
}

/// Mutable state executing a [`GraphProgram`]: the witness tape, the
/// input cursor, and the workers' done flags. Reusable across proofs — `run`
/// stamps `flags` with a fresh `phase` instead of zeroing them, and the tape's
/// compute cells are simply overwritten.
///
/// Sized against a specific [`GraphProgram`]; pairing this with the wrong
/// program is a bug (asserted in [`GraphExecutor::new`]).
pub struct GraphExecutorState {
    /// Layout: `[advice | lookups | consts]`.
    tape: Vec<Fr>,
    input_cursor: usize,
    /// Per-instruction done flag; stamped with the current `phase` on
    /// completion. Workers spin-wait on parent flags before executing.
    flags: Vec<AtomicU8>,
    /// Wrapping run counter that stamps `flags`. `run` is synchronous
    /// (workers join before return), so wrapping is safe.
    phase: u8,
}

/// Ephemeral binding of a [`GraphProgram`] with a [`GraphExecutorState`],
/// implementing [`PopulateInputs`] so `load_proof_wire` can stream witnesses
/// into the tape and driving [`Self::run`] for the compute phase.
pub struct GraphExecutor<'a> {
    pub program: &'a GraphProgram,
    pub state: &'a mut GraphExecutorState,
}

impl GraphProgram {
    /// Records `circuit`'s full populate trace (constraints + onion-commit
    /// pin + PV extraction) into a graph IR and lowers it. Any valid proof
    /// for the static shape works as `representative_proof`.
    pub fn new(
        circuit: &StaticVerifierCircuit,
        lookup_bits: usize,
        representative_proof: &Proof<RootConfig>,
    ) -> Self {
        let mut ir = Halo2IRBuilder::new(lookup_bits);
        let pvs_wire = circuit.populate_pvs(&mut ir, representative_proof);
        let pv_offsets = pvs_wire
            .to_vec()
            .iter()
            .map(|cell| match cell {
                GraphCell::Cell(_, offset, _) => *offset,
                GraphCell::Const(_) => unreachable!("public value must be an advice cell"),
            })
            .collect();
        let mut program = Self::lower(ir);
        program.pv_offsets = pv_offsets;
        program
    }

    /// Lowers a recorded IR into executable form (with no public values).
    fn lower(ir: Halo2IRBuilder) -> Self {
        let advice_cells = ir.total_ctx_len();
        let lookup_cells = ir.total_lookups_len();
        let lookup_bits = ir.lookup_bits();
        let const_base = advice_cells + lookup_cells;

        let mut consts: Vec<Fr> = Vec::new();
        let mut const_map: HashMap<[u8; 32], u32> = HashMap::new();
        let mut operand_offsets: Vec<u32> = Vec::new();
        let mut operand_bits: Vec<u16> = Vec::new();
        let mut const_inds: Vec<u32> = Vec::new();
        let mut input_insts: Vec<GraphCoreInst> = Vec::new();
        let mut compute: Vec<(u32, GraphCoreInst)> = Vec::with_capacity(ir.nodes.len());

        for (node, meta) in ir.nodes.iter().zip(&ir.node_meta) {
            let arg_lo = operand_offsets.len() as u32;
            for (cell, &arg_offset) in node.operands.iter().zip(&meta.arg_offsets) {
                let offset = match cell {
                    GraphCell::Cell(..) => {
                        assert_ne!(
                            arg_offset, UNMATERIALIZED,
                            "cell operand must be materialized"
                        );
                        arg_offset
                    }
                    GraphCell::Const(value) => {
                        let idx = *const_map.entry(value.to_bytes()).or_insert_with(|| {
                            consts.push(*value);
                            (consts.len() - 1) as u32
                        });
                        const_base + idx as usize
                    }
                };
                operand_offsets.push(u32::try_from(offset).expect("tape offset exceeds u32"));
                operand_bits.push(cell.bits() as u16);
            }
            let ci_lo = const_inds.len() as u32;
            const_inds.extend_from_slice(&meta.constant_skip_inds);
            let inst = GraphCoreInst {
                opcode: node.opcode,
                args: (arg_lo, operand_offsets.len() as u32),
                const_inds: (ci_lo, const_inds.len() as u32),
                ctx_offset: meta.ctx_offset as u32,
                ctx_len: meta.ctx_len as u32,
                lookup_offset: meta.lookup_offset as u32,
                lookups_len: meta.lookups_len as u32,
                // Filled in later, after `insts` is level-sorted and cell → inst
                // resolution is available.
                dep_list: (0, 0),
            };
            match node.opcode {
                Halo2Opcode::LoadWitness => input_insts.push(inst),
                _ => compute.push((meta.level, inst)),
            }
        }

        // Two orders coexist from here on:
        // - `insts` (and the parallel `flags`) is LEVEL-sorted: workers claim instructions off the
        //   atomic cursor in this order, so claims are roughly topological and parent spin-waits
        //   stay short. The sort is stable, keeping emission order within a level (write locality).
        // - `release_order` (built below) stays in EMISSION order, which is what the release walk
        //   needs.
        // `order[sorted]` is the emission index of the instruction placed at
        // sorted position `sorted`; `emission_to_sorted` is its inverse,
        // translating an emission index into the `insts`/`flags` slot.
        let mut order: Vec<u32> = (0..compute.len() as u32).collect();
        order.sort_by_key(|&i| compute[i as usize].0);
        let mut insts: Vec<GraphCoreInst> = order.iter().map(|&i| compute[i as usize].1).collect();
        let mut emission_to_sorted = vec![0u32; compute.len()];
        for (sorted, &emission) in order.iter().enumerate() {
            emission_to_sorted[emission as usize] = sorted as u32;
        }

        // In emission order both write offsets are monotone (bump-cursor
        // allocation), so writers' disjoint write ranges tile each tape in
        // order; the advice tape additionally has input-populated gaps
        // between them. Walking writers in emission order lets the release
        // walk fold each gap into the pending range before it. Each entry
        // still needs `emission_to_sorted` to name the writer's flag slot,
        // since flags are stamped at level-sorted positions.
        let mut release_order: Vec<ReleaseEntry> = Vec::with_capacity(insts.len());
        let (mut prev_a, mut prev_l) = (0u32, 0u32);
        for (i, (_, inst)) in compute.iter().enumerate() {
            if inst.ctx_len == 0 && inst.lookups_len == 0 {
                continue;
            }
            debug_assert!(inst.ctx_offset >= prev_a && inst.lookup_offset >= prev_l);
            prev_a = inst.ctx_offset + inst.ctx_len;
            prev_l = inst.lookup_offset + inst.lookups_len;
            release_order.push(ReleaseEntry {
                inst: emission_to_sorted[i],
                advice_end: prev_a,
                lookup_end: prev_l,
            });
        }
        drop(compute);

        // Build the dependency graph: for each compute inst, list the indices
        // of parent compute insts (whose output cells this inst reads as
        // operands). Input-inst parents are dropped — their cells are already
        // populated before `run` starts. Since `insts` is level-sorted and
        // dependencies point to strictly-lower levels, parent indices are
        // always strictly less than the child's index.
        let mut cell_to_compute_inst: Vec<i32> = vec![-1; advice_cells];
        for (idx, inst) in insts.iter().enumerate() {
            let ctx_lo = inst.ctx_offset as usize;
            let ctx_hi = ctx_lo + inst.ctx_len as usize;
            cell_to_compute_inst[ctx_lo..ctx_hi].fill(idx as i32);
        }
        let mut dep_inds: Vec<u32> = Vec::new();
        let mut local_deps: Vec<u32> = Vec::new();
        for inst in &mut insts {
            local_deps.clear();
            let (arg_lo, arg_hi) = inst.args;
            for &arg_offset in &operand_offsets[arg_lo as usize..arg_hi as usize] {
                let offset = arg_offset as usize;
                if offset < advice_cells {
                    let parent = cell_to_compute_inst[offset];
                    if parent >= 0 {
                        let parent = parent as u32;
                        if !local_deps.contains(&parent) {
                            local_deps.push(parent);
                        }
                    }
                }
            }
            let dep_lo = dep_inds.len() as u32;
            dep_inds.extend_from_slice(&local_deps);
            let dep_hi = dep_inds.len() as u32;
            inst.dep_list = (dep_lo, dep_hi);
        }
        drop(cell_to_compute_inst);

        GraphProgram {
            advice_cells,
            lookup_cells,
            lookup_bits,
            consts,
            operand_offsets,
            operand_bits,
            const_inds,
            insts,
            input_insts,
            release_order,
            dep_inds,
            pv_offsets: Vec::new(),
        }
    }

    /// Total number of advice-tape cells written per run.
    pub fn advice_cells(&self) -> usize {
        self.advice_cells
    }

    /// Total number of range-check tape cells written per run.
    pub fn lookup_cells(&self) -> usize {
        self.lookup_cells
    }

    /// Advice offsets of the circuit's public values, in instance order.
    pub fn pv_offsets(&self) -> &[usize] {
        &self.pv_offsets
    }
}

impl GraphExecutorState {
    /// Allocates a state buffer sized against `program`. Reusable for any
    /// proof of `program`'s static shape.
    pub fn new(program: &GraphProgram) -> Self {
        let const_base = program.advice_cells + program.lookup_cells;
        let mut tape = vec![Fr::ZERO; const_base + program.consts.len()];
        tape[const_base..].copy_from_slice(&program.consts);
        let flags: Vec<AtomicU8> = (0..program.insts.len()).map(|_| AtomicU8::new(0)).collect();
        Self {
            tape,
            input_cursor: 0,
            flags,
            phase: 0,
        }
    }

    /// Rewinds the input cursor for a new proof; compute cells are overwritten
    /// by the next [`GraphExecutor::run`], so no other state needs clearing.
    pub fn reset(&mut self) {
        self.input_cursor = 0;
    }

    /// The advice (context) tape; matches `Context::advice_cells()` of the halo2
    /// backend after [`GraphExecutor::run`].
    pub fn advice(&self, program: &GraphProgram) -> &[Fr] {
        &self.tape[..program.advice_cells]
    }

    /// The range-check tape; matches the values sent to `add_cell_to_lookup`.
    pub fn lookups(&self, program: &GraphProgram) -> &[Fr] {
        &self.tape[program.advice_cells..program.advice_cells + program.lookup_cells]
    }
}

impl<'a> GraphExecutor<'a> {
    pub fn new(program: &'a GraphProgram, state: &'a mut GraphExecutorState) -> Self {
        assert_eq!(
            state.tape.len(),
            program.advice_cells + program.lookup_cells + program.consts.len(),
            "state was allocated against a different program"
        );
        assert_eq!(
            state.flags.len(),
            program.insts.len(),
            "state was allocated against a different program"
        );
        Self { program, state }
    }

    pub fn program(&self) -> &GraphProgram {
        self.program
    }

    /// See [`GraphExecutorState::advice`].
    pub fn advice(&self) -> &[Fr] {
        self.state.advice(self.program)
    }

    /// See [`GraphExecutorState::lookups`].
    pub fn lookups(&self) -> &[Fr] {
        self.state.lookups(self.program)
    }

    /// Replays the next `LoadWitness` with `value` and returns its advice offset.
    fn populate_input(&mut self, expected: Halo2Opcode, value: Fr) -> usize {
        debug_assert!(matches!(expected, Halo2Opcode::LoadWitness));
        let inst = *self
            .program
            .input_insts
            .get(self.state.input_cursor)
            .expect("more input loads than recorded input instructions");
        debug_assert_eq!(
            inst.opcode, expected,
            "input load {} kind mismatch",
            self.state.input_cursor
        );
        self.state.input_cursor += 1;
        let advice = &mut self.state.tape[..self.program.advice_cells];
        advice[inst.ctx_offset as usize] = value;
        debug_assert!(inst.ctx_len == 1);
        debug_assert!(inst.lookups_len == 0);
        inst.ctx_offset as usize
    }

    /// Evaluates the compute schedule with `num_threads` workers, barrier-free:
    /// workers claim instructions off an atomic cursor, spin-wait on parents'
    /// done flags, execute, and Release-store their own flag. Flags are
    /// stamped with `phase` (not zeroed between runs).
    ///
    /// Meanwhile the calling thread streams newly-materialized tape ranges
    /// through `on_delta(advice_offset, advice_delta, lookup_offset,
    /// lookup_delta)`: writers are polled in the program's release order
    /// (emission order, ascending tape offsets) with a bounded spin, so
    /// ready writers extend one pending block that also absorbs the
    /// input-populated gap cells between them. The block flushes once its
    /// advice span reaches `MAX_FLUSH_CELLS` or when a slow writer defers to
    /// the retry passes; sub-`MIN_FLUSH_CELLS` blocks are deferred for
    /// re-merging instead of flushed as tiny H2D copies.
    #[allow(unsafe_code)]
    pub fn run<F>(&mut self, num_threads: usize, mut on_delta: F)
    where
        F: FnMut(usize, &[Fr], usize, &[Fr]),
    {
        assert!(num_threads > 0);
        assert_eq!(
            self.state.input_cursor,
            self.program.input_insts.len(),
            "all proof inputs must be populated before run"
        );

        // Bump phase; skip 0 so a fresh executor never reports zeroed flags as
        // ready.
        self.state.phase = self.state.phase.wrapping_add(1);
        if self.state.phase == 0 {
            self.state.phase = 1;
        }
        let phase = self.state.phase;

        // Detach the tape so worker threads share the program and flags
        // without aliasing the buffer.
        let mut tape = std::mem::take(&mut self.state.tape);
        let tape_ptr = TapePtr(tape.as_mut_ptr());
        let program: &GraphProgram = self.program;
        let flags: &[AtomicU8] = &self.state.flags;
        let advice_cells = program.advice_cells;
        let lookup_cells = program.lookup_cells;
        let n_insts = program.insts.len();
        // Level-sorted claims are roughly topological, so parent spin-waits stay short.
        let claim_cursor = AtomicUsize::new(0);

        // A panicked worker would leave its flag unset forever, livelocking
        // peer spin-waits. Every unbounded wait checks `poisoned` and panics.
        let poisoned = AtomicBool::new(false);
        struct PoisonOnPanic<'a>(&'a AtomicBool);
        impl Drop for PoisonOnPanic<'_> {
            fn drop(&mut self) {
                if std::thread::panicking() {
                    self.0.store(true, Ordering::Relaxed);
                }
            }
        }

        std::thread::scope(|s| {
            for _ in 0..num_threads {
                let claim_cursor = &claim_cursor;
                let poisoned = &poisoned;
                s.spawn(move || {
                    let _poison = PoisonOnPanic(poisoned);
                    let mut args: Vec<Fr> = Vec::new();
                    let mut bits: Vec<u16> = Vec::new();
                    loop {
                        let idx = claim_cursor.fetch_add(1, Ordering::Relaxed);
                        if idx >= n_insts {
                            break;
                        }
                        let inst = &program.insts[idx];
                        // Spin-wait until every parent's flag matches `phase`.
                        let (dep_lo, dep_hi) = inst.dep_list;
                        for d in dep_lo as usize..dep_hi as usize {
                            let parent = program.dep_inds[d] as usize;
                            while flags[parent].load(Ordering::Acquire) != phase {
                                assert!(
                                    !poisoned.load(Ordering::Relaxed),
                                    "graph executor worker panicked"
                                );
                                std::hint::spin_loop();
                            }
                        }
                        program.eval_inst(inst, tape_ptr, &mut args, &mut bits);
                        // Release publishes tape writes to Acquire loaders of the flag.
                        flags[idx].store(phase, Ordering::Release);
                    }
                });
            }

            // Emission-order release walk on the calling thread (see the `run` doc).
            {
                /// Flag polls before deferring a writer to the retry list.
                const MAX_SPIN_TRIES: usize = 32;
                /// Ranges smaller than this are deferred for re-merging
                /// instead of flushed as tiny H2D copies.
                const MIN_FLUSH_CELLS: u32 = 8 * 1024;
                /// Pending ranges flush once their advice span reaches this
                /// size so long runs stream incrementally.
                const MAX_FLUSH_CELLS: u32 = 1 << 20;
                /// `inst` marker for an already-materialized deferred range.
                const READY_SENTINEL: u32 = u32::MAX;

                /// Contiguous spans on the advice and lookup tapes.
                #[derive(Clone, Copy)]
                struct Range {
                    a_start: u32,
                    a_end: u32,
                    l_start: u32,
                    l_end: u32,
                }
                impl Range {
                    /// An empty range positioned at (`a`, `l`).
                    fn empty_at(a: u32, l: u32) -> Self {
                        Range {
                            a_start: a,
                            a_end: a,
                            l_start: l,
                            l_end: l,
                        }
                    }
                    fn cells(&self) -> u32 {
                        (self.a_end - self.a_start) + (self.l_end - self.l_start)
                    }
                }
                /// A `Range` gated on `flags[inst]` (or `READY_SENTINEL`).
                type Entry = (u32, Range);

                let try_wait = |inst: u32| {
                    if inst == READY_SENTINEL {
                        return true;
                    }
                    let flag = &flags[inst as usize];
                    for _ in 0..MAX_SPIN_TRIES {
                        if flag.load(Ordering::Acquire) == phase {
                            return true;
                        }
                        std::hint::spin_loop();
                    }
                    false
                };
                // Fires `on_delta`, unless the range is sub-`MIN_FLUSH_CELLS`
                // and `defer_to` is given (then queued as a ready entry).
                //
                // Safety: flushed ranges are unions of disjoint release
                // ranges whose flags passed `try_wait`; the Acquire load
                // pairs with each worker's Release store, publishing the
                // writes. Gap cells folded into a release range are
                // input-populated before `run` by this thread, so program
                // order suffices for them.
                let mut flush_or_defer = |r: Range, defer_to: Option<&mut Vec<Entry>>| {
                    if r.cells() == 0 {
                        return;
                    }
                    if let Some(defer) = defer_to {
                        if r.cells() < MIN_FLUSH_CELLS {
                            defer.push((READY_SENTINEL, r));
                            return;
                        }
                    }
                    let advice_delta: &[Fr] = unsafe {
                        std::slice::from_raw_parts(
                            (tape_ptr.0 as *const Fr).add(r.a_start as usize),
                            (r.a_end - r.a_start) as usize,
                        )
                    };
                    let lookup_delta: &[Fr] = unsafe {
                        std::slice::from_raw_parts(
                            (tape_ptr.0 as *const Fr).add(advice_cells + r.l_start as usize),
                            (r.l_end - r.l_start) as usize,
                        )
                    };
                    on_delta(
                        r.a_start as usize,
                        advice_delta,
                        r.l_start as usize,
                        lookup_delta,
                    );
                };

                // Pass 0: poll writers in emission order, so each ready
                // writer extends `pend` (absorbing the input-populated gap
                // cells before its write range). A slow writer flushes the
                // block and queues a gap-covering range for retry.
                let mut failed: Vec<Entry> = Vec::new();
                let mut pend = Range::empty_at(0, 0);
                for &ReleaseEntry {
                    inst,
                    advice_end,
                    lookup_end,
                } in &program.release_order
                {
                    if try_wait(inst) {
                        pend.a_end = advice_end;
                        pend.l_end = lookup_end;
                        if pend.a_end - pend.a_start >= MAX_FLUSH_CELLS {
                            flush_or_defer(pend, None);
                            pend = Range::empty_at(advice_end, lookup_end);
                        }
                    } else {
                        flush_or_defer(pend, Some(&mut failed));
                        failed.push((
                            inst,
                            Range {
                                a_start: pend.a_end,
                                a_end: advice_end,
                                l_start: pend.l_end,
                                l_end: lookup_end,
                            },
                        ));
                        pend = Range::empty_at(advice_end, lookup_end);
                    }
                }
                // Trailing input-populated cells belong to the final range.
                pend.a_end = advice_cells as u32;
                pend.l_end = lookup_cells as u32;
                if failed.is_empty() {
                    flush_or_defer(pend, None);
                } else {
                    flush_or_defer(pend, Some(&mut failed));
                }

                // Retry passes: re-poll deferred writers (still in emission
                // order) with the same merge rule until all have landed.
                // Deferral stops once every entry is ready, so the final
                // pass flushes everything (termination).
                while !failed.is_empty() {
                    assert!(
                        !poisoned.load(Ordering::Relaxed),
                        "graph executor worker panicked"
                    );
                    let allow_defer = failed.iter().any(|&(idx, _)| idx != READY_SENTINEL);
                    let mut still: Vec<Entry> = Vec::new();
                    let mut pend = Range::empty_at(0, 0);
                    for &(idx, r) in &failed {
                        if try_wait(idx) {
                            if pend.a_end == r.a_start && pend.l_end == r.l_start {
                                pend.a_end = r.a_end;
                                pend.l_end = r.l_end;
                            } else {
                                flush_or_defer(
                                    pend,
                                    if allow_defer { Some(&mut still) } else { None },
                                );
                                pend = r;
                            }
                            if pend.a_end - pend.a_start >= MAX_FLUSH_CELLS {
                                flush_or_defer(pend, None);
                                pend = Range::empty_at(pend.a_end, pend.l_end);
                            }
                        } else {
                            flush_or_defer(pend, Some(&mut still));
                            still.push((idx, r));
                            pend = Range::empty_at(r.a_end, r.l_end);
                        }
                    }
                    if allow_defer && !still.is_empty() {
                        flush_or_defer(pend, Some(&mut still));
                    } else {
                        flush_or_defer(pend, None);
                    }
                    failed = still;
                }
            }
        });

        self.state.tape = tape;
    }
}

impl GraphProgram {
    #[allow(unsafe_code)]
    fn eval_inst(
        &self,
        inst: &GraphCoreInst,
        tape: TapePtr,
        args: &mut Vec<Fr>,
        bits: &mut Vec<u16>,
    ) {
        args.clear();
        bits.clear();
        let (lo, hi) = inst.args;
        for i in lo as usize..hi as usize {
            let offset = self.operand_offsets[i] as usize;
            // Safety: operands are input/const (prefilled) or parent outputs
            // published via the parent-flag Acquire before this call.
            args.push(unsafe { *tape.0.add(offset) });
            bits.push(self.operand_bits[i]);
        }
        // Safety: each instruction's ctx/lookup ranges are disjoint from all
        // others', so these exclusive slices never overlap across threads.
        let ctx = unsafe {
            std::slice::from_raw_parts_mut(
                tape.0.add(inst.ctx_offset as usize),
                inst.ctx_len as usize,
            )
        };
        let lookups = unsafe {
            std::slice::from_raw_parts_mut(
                tape.0.add(self.advice_cells + inst.lookup_offset as usize),
                inst.lookups_len as usize,
            )
        };
        let (ci_lo, ci_hi) = inst.const_inds;
        interpret_op(
            &inst.opcode,
            args,
            bits,
            ctx,
            lookups,
            self.lookup_bits,
            &self.const_inds[ci_lo as usize..ci_hi as usize],
        );
    }
}

impl<'a> ChipBase for GraphExecutor<'a> {
    /// Wires are absolute offsets into the executor's tape.
    type F = usize;
}

impl<'a> PopulateInputs for GraphExecutor<'a> {
    fn load_witness(&mut self, value: Fr) -> usize {
        self.populate_input(Halo2Opcode::LoadWitness, value)
    }

    fn bb_load_reduced_witness(&mut self, value: BabyBear) -> ReducedBabyBearWire<usize> {
        let offset =
            self.populate_input(Halo2Opcode::LoadWitness, Fr::from(value.as_canonical_u64()));
        ReducedBabyBearWire::assume_reduced(BabyBearWire {
            value: offset,
            max_bits: BABYBEAR_MAX_BITS,
        })
    }

    fn ext_load_reduced_witness(&mut self, value: BabyBearExt4) -> ReducedBabyBearExt4Wire<usize> {
        let coeffs = value.as_basis_coefficients_slice();
        ReducedBabyBearExt4Wire::assume_reduced(core::array::from_fn(|i| {
            self.bb_load_reduced_witness(coeffs[i])
        }))
    }
}

/// Streams graph-executor tape deltas into advice columns.
///
/// Column storage is [`AdviceColumns<Fr>`] (`DeviceBuffer`s under `halo2-gpu`,
/// `Vec`s otherwise); placement is a pure function of the tape offset so
/// disjoint deltas may arrive in any order. Layout mirrors
/// `PagedWitnessContext::push_advice` (gate columns split at pinned break
/// points, break value duplicated at row 0 of the next column) and
/// `BaseCircuitBuilder::assign_lookups_in_phase` (lookup columns round-robin:
/// value `i` at column `i % L`, row `i / L`).
///
/// Columns are zero-filled and allocated on the first [`Self::append`], then
/// segments write directly into their row range — no intermediate host buffer.
pub struct FusedColumnBuilder {
    // ---- Config (immutable after `new`) ------------------------------------
    n: usize,
    num_advice_columns: usize,
    /// Pinned break rows, indexed by gate column.
    break_points: Vec<usize>,
    /// Absolute advice offset of row 0 of each gate column;
    /// `col_starts[c + 1] = col_starts[c] + break_points[c]` (row 0 of column
    /// `c+1` duplicates column `c`'s break-row value).
    col_starts: Vec<usize>,
    /// Physical column indices of the range-check lookup advice columns.
    lookup_col_indices: Vec<usize>,

    /// Advice columns (lazily allocated on first `append`).
    columns: AdviceColumns<Fr>,
}

impl FusedColumnBuilder {
    pub fn new(
        n: usize,
        num_advice_columns: usize,
        break_points: Vec<usize>,
        lookup_col_indices: Vec<usize>,
    ) -> Self {
        let mut col_starts = Vec::with_capacity(break_points.len() + 1);
        col_starts.push(0usize);
        for &bp in &break_points {
            col_starts.push(col_starts.last().unwrap() + bp);
        }
        Self {
            n,
            num_advice_columns,
            break_points,
            col_starts,
            lookup_col_indices,
            columns: AdviceColumns::<Fr>::new(),
        }
    }

    fn ensure_allocated(&mut self) {
        if !self.columns.is_empty() {
            return;
        }
        self.columns.reserve_exact(self.num_advice_columns);
        for _ in 0..self.num_advice_columns {
            #[cfg(feature = "halo2-gpu")]
            {
                let buf: DeviceBuffer<Fr> =
                    DeviceBuffer::<Fr>::with_capacity_on(self.n, &HALO2_GPU_CTX);
                buf.fill_zero_on(&HALO2_GPU_CTX)
                    .expect("zero-fill advice column");
                self.columns.push(buf);
            }
            #[cfg(not(feature = "halo2-gpu"))]
            self.columns.push(vec![Fr::ZERO; self.n]);
        }
    }

    /// Writes `advice_delta` (starting at `advice_offset`) and `lookup_delta`
    /// (starting at `lookup_offset`) into the advice columns. Placement is a
    /// pure function of the offsets, so disjoint deltas may arrive in any order.
    pub fn append(
        &mut self,
        advice_offset: usize,
        advice_delta: &[Fr],
        lookup_offset: usize,
        lookup_delta: &[Fr],
    ) {
        self.ensure_allocated();

        // --- Gate stream: contiguous write per (column, row-range) segment.
        //
        // Column `c` covers `[col_starts[c], col_starts[c+1]]` inclusive; the
        // shared endpoint (break value) is duplicated at `(c, break_points[c])`
        // and `(c+1, 0)`. On crossing a break, `delta_pos -= 1` re-emits the
        // break value as row 0 of the next column.
        if !advice_delta.is_empty() {
            let c = self.col_starts.partition_point(|&s| s <= advice_offset) - 1;
            let (mut col, mut row) = if c > 0 && advice_offset == self.col_starts[c] {
                (c - 1, self.break_points[c - 1])
            } else {
                (c, advice_offset - self.col_starts[c])
            };
            let mut delta_pos = 0usize;
            while delta_pos < advice_delta.len() {
                let cur_break_point = self.break_points.get(col).copied();
                let rows_until_break = match cur_break_point {
                    Some(bp) => {
                        debug_assert!(bp >= row);
                        bp - row + 1
                    }
                    None => usize::MAX,
                };
                let delta_remaining = advice_delta.len() - delta_pos;
                let take = rows_until_break.min(delta_remaining);
                let src = &advice_delta[delta_pos..delta_pos + take];
                #[cfg(feature = "halo2-gpu")]
                self.columns[col]
                    .mut_slice(row..row + take)
                    .copy_from_host(src, &HALO2_GPU_CTX)
                    .expect("H2D advice gate segment");
                #[cfg(not(feature = "halo2-gpu"))]
                self.columns[col][row..row + take].copy_from_slice(src);
                delta_pos += take;
                if cur_break_point.is_some() && take == rows_until_break {
                    col += 1;
                    row = 0;
                    delta_pos -= 1; // Re-emit the break value as row 0 of the new column.
                } else {
                    row += take;
                }
            }
        }

        // --- Lookup stream: the value at global lookup index `g`
        // (`lookup_offset` + its position in the delta) lands at column
        // `lookup_col_indices[g % l]`, row `g / l`. On GPU, a column's values
        // sit at delta positions `first_pos, first_pos + l, ...` and their
        // rows are consecutive (each stride step advances `g` by exactly
        // `l`), so gather the stride into a host buffer and issue one
        // contiguous H2D copy per column. On host, just scatter.
        let l = self.lookup_col_indices.len();
        #[cfg(feature = "halo2-gpu")]
        if !lookup_delta.is_empty() {
            for col in 0..l {
                // First delta position `i` with `(lookup_offset + i) % l == col`.
                let first_pos = (col + l - lookup_offset % l) % l;
                if first_pos >= lookup_delta.len() {
                    continue;
                }
                let num_values = (lookup_delta.len() - first_pos).div_ceil(l);
                let first_row = (lookup_offset + first_pos) / l;
                let host_buf: Vec<Fr> = (0..num_values)
                    .map(|i| lookup_delta[first_pos + i * l])
                    .collect();
                self.columns[self.lookup_col_indices[col]]
                    .mut_slice(first_row..first_row + num_values)
                    .copy_from_host(&host_buf, &HALO2_GPU_CTX)
                    .expect("H2D lookup column gather");
            }
        }
        #[cfg(not(feature = "halo2-gpu"))]
        for (i, v) in lookup_delta.iter().enumerate() {
            let global = lookup_offset + i;
            let col = self.lookup_col_indices[global % l];
            let row = global / l;
            self.columns[col][row] = *v;
        }
    }

    /// Consumes the advice columns, leaving the builder empty.
    pub fn take_columns(&mut self) -> AdviceColumns<Fr> {
        assert!(
            !self.columns.is_empty(),
            "take_columns: no data was ever appended",
        );
        std::mem::take(&mut self.columns)
    }

    /// Diagnostic-only D2H copy of every device column, for byte-comparing
    /// against the legacy `BaseCircuitBuilder` path. Not on the hot path.
    #[cfg(all(feature = "halo2-gpu", test))]
    pub fn snapshot_columns_to_host(&self) -> Vec<Vec<Fr>> {
        use openvm_cuda_common::copy::MemCopyD2H;
        self.columns
            .iter()
            .map(|d| d.to_host_on(&HALO2_GPU_CTX).expect("D2H advice column"))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    #[cfg(feature = "halo2-gpu")]
    use std::time::{Duration, Instant};

    use halo2_base::gates::{
        circuit::{builder::BaseCircuitBuilder, CircuitBuilderStage},
        RangeChip,
    };
    use openvm_stark_sdk::{
        config::baby_bear_bn254_poseidon2::{
            BabyBearBn254Poseidon2Config as RootConfig, BabyBearBn254Poseidon2CpuEngine,
        },
        openvm_stark_backend::{
            proof::Proof,
            test_utils::{test_system_params_small, InteractionsFixture11, TestFixture},
            StarkEngine,
        },
    };
    #[cfg(feature = "halo2-gpu")]
    use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

    use super::*;
    use crate::{
        backend::Halo2Backend,
        stages::{full_pipeline::load_proof_wire, proof_shape::log_heights_per_air_from_proof},
        StaticVerifierCircuit,
    };
    #[cfg(feature = "halo2-gpu")]
    use crate::{
        test_fixtures::{fixture_circuit_and_proof, FIXTURE_K},
        Halo2Params, StaticVerifierProvingKey, StaticVerifierShape,
    };

    const K: usize = 22;
    const LOOKUP_BITS: usize = K - 1;

    /// Flattens the range tape: every value sent to `add_cell_to_lookup`, in order.
    fn lookup_tape(range: &RangeChip<Fr>) -> Vec<Fr> {
        let map = range.lookup_manager()[0].cells_to_lookup.lock().unwrap();
        assert!(map.len() <= 1, "expected a single context tag");
        map.values()
            .flat_map(|cells| cells.iter().map(|c| c[0].value.evaluate()))
            .collect()
    }

    fn build_and_run(
        circuit: &StaticVerifierCircuit,
        proof: &Proof<RootConfig>,
        log_heights_per_air: &[usize],
        num_threads: usize,
    ) -> (GraphProgram, GraphExecutorState) {
        let mut ir = Halo2IRBuilder::new(LOOKUP_BITS);
        circuit.populate_verify_stark_constraints(&mut ir, proof);
        let program = GraphProgram::lower(ir);
        let mut state = GraphExecutorState::new(&program);
        {
            let mut executor = GraphExecutor::new(&program, &mut state);
            load_proof_wire(&mut executor, proof, log_heights_per_air);

            // Rebuild both tapes from the streamed deltas: every cell must be
            // covered and end at the tape's final value.
            let mut shadow_advice = vec![Fr::ZERO; executor.advice().len()];
            let mut covered_advice = vec![false; executor.advice().len()];
            let mut shadow_lookups = vec![Fr::ZERO; executor.lookups().len()];
            let mut covered_lookups = vec![false; executor.lookups().len()];
            executor.run(num_threads, |a_off, advice, l_off, lookups| {
                shadow_advice[a_off..a_off + advice.len()].copy_from_slice(advice);
                covered_advice[a_off..a_off + advice.len()].fill(true);
                shadow_lookups[l_off..l_off + lookups.len()].copy_from_slice(lookups);
                covered_lookups[l_off..l_off + lookups.len()].fill(true);
            });
            assert!(covered_advice.iter().all(|&c| c), "uncovered advice cells");
            assert!(covered_lookups.iter().all(|&c| c), "uncovered lookup cells");
            assert_eq!(shadow_advice, executor.advice(), "advice deltas");
            assert_eq!(shadow_lookups, executor.lookups(), "lookup deltas");
        }
        (program, state)
    }

    #[test]
    fn graph_executor_matches_halo2_backend() {
        let engine: BabyBearBn254Poseidon2CpuEngine =
            BabyBearBn254Poseidon2CpuEngine::new(test_system_params_small(2, 8, 3));
        let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
        let log_heights_per_air = log_heights_per_air_from_proof(&proof);
        let circuit = StaticVerifierCircuit::try_new(vk, Default::default(), &log_heights_per_air)
            .expect("static circuit params");

        // Reference: the real halo2 population.
        let mut builder = BaseCircuitBuilder::from_stage(CircuitBuilderStage::Mock)
            .use_k(K)
            .use_lookup_bits(LOOKUP_BITS);
        let range = Arc::new(builder.range_chip());
        let ctx = builder.main(0);
        let mut backend = Halo2Backend::new(range.clone(), ctx);
        circuit.populate_verify_stark_constraints(&mut backend, &proof);
        let real_advice: Vec<Fr> = backend
            .ctx_mut()
            .advice
            .iter()
            .map(|a| a.evaluate())
            .collect();
        let real_lookups = lookup_tape(&range);

        let (program, state) = build_and_run(&circuit, &proof, &log_heights_per_air, 4);
        assert_eq!(
            state.advice(&program).len(),
            real_advice.len(),
            "advice len"
        );
        assert_eq!(state.advice(&program), &real_advice[..], "advice tape");
        assert_eq!(state.lookups(&program), &real_lookups[..], "range tape");

        // Determinism across schedules: single-threaded run matches.
        let (seq_program, seq_state) = build_and_run(&circuit, &proof, &log_heights_per_air, 1);
        assert_eq!(seq_state.advice(&seq_program), state.advice(&program));
        assert_eq!(seq_state.lookups(&seq_program), state.lookups(&program));
    }

    /// Runs the executor + fused H2D copies; returns wall time.
    #[cfg(feature = "halo2-gpu")]
    fn timed_run(
        executor: &mut GraphExecutor<'_>,
        builder: &mut FusedColumnBuilder,
        num_threads: usize,
    ) -> Duration {
        let start = Instant::now();
        executor.run(
            num_threads,
            |advice_offset, advice, lookup_offset, lookups| {
                builder.append(advice_offset, advice, lookup_offset, lookups)
            },
        );
        start.elapsed()
    }

    /// Production-path setup: STARK-prove the root-shaped fixture, then
    /// `keygen` against an in-memory SRS to get a real pinning.
    #[cfg(feature = "halo2-gpu")]
    fn keygen_fixture_static_verifier() -> (StaticVerifierProvingKey, Proof<RootConfig>) {
        let (circuit, proof) = fixture_circuit_and_proof();
        let shape = StaticVerifierShape {
            k: FIXTURE_K,
            lookup_bits: FIXTURE_K - 1,
            minimum_rows: 20,
            instance_columns: 1,
        };

        let start = Instant::now();
        let params = Halo2Params::setup(FIXTURE_K as u32, ChaCha20Rng::seed_from_u64(42));
        println!("SRS setup (k={FIXTURE_K}): {:?}", start.elapsed());

        let start = Instant::now();
        let pk = StaticVerifierProvingKey::keygen(&params, shape, circuit, &proof);
        println!("static verifier keygen: {:?}", start.elapsed());
        (pk, proof)
    }

    #[test]
    #[cfg(feature = "halo2-gpu")]
    #[ignore = "requires CUDA GPU; slow (fixture STARK prove + halo2 keygen)"]
    fn graph_executor_root_proof() {
        use halo2_base::{
            gates::circuit::MaybeRangeConfig,
            halo2_proofs::{halo2curves::bn256::G1Affine, plonk::create_constraint_system},
        };

        let (pk, proof) = keygen_fixture_static_verifier();
        let metadata = &pk.pinning.metadata;
        let log_heights_per_air = log_heights_per_air_from_proof(&proof);

        // Physical column layout for the FusedColumnBuilder
        // (mirrors `StaticVerifierProvingKey::generate_witness`).
        let n = 1usize << metadata.config_params.k;
        let (cs, config) = create_constraint_system::<G1Affine, BaseCircuitBuilder<Fr>>(
            metadata.config_params.clone(),
        );
        let num_advice_columns = cs.num_advice_columns();
        let MaybeRangeConfig::WithRange(range_config) = &config.base else {
            panic!("static verifier requires lookup advice columns");
        };
        let lookup_col_indices: Vec<usize> = range_config.lookup_advice[0]
            .iter()
            .map(|c| c.index())
            .collect();
        let break_points = metadata.break_points[0].clone();
        let fused_builder = || {
            FusedColumnBuilder::new(
                n,
                num_advice_columns,
                break_points.clone(),
                lookup_col_indices.clone(),
            )
        };

        let start = Instant::now();
        let mut ir = Halo2IRBuilder::new(pk.shape.lookup_bits);
        pk.circuit.populate_pvs(&mut ir, &proof);
        println!("IR build: {:?}", start.elapsed());

        let start = Instant::now();
        let program = GraphProgram::lower(ir);
        let mut state = GraphExecutorState::new(&program);
        println!(
            "lowering: {:?} ({} insts)",
            start.elapsed(),
            program.insts.len()
        );

        let start = Instant::now();
        {
            let mut executor = GraphExecutor::new(&program, &mut state);
            load_proof_wire(&mut executor, &proof, &log_heights_per_air);
        }
        println!("input population: {:?}", start.elapsed());

        let mut builder = fused_builder();
        let total = timed_run(
            &mut GraphExecutor::new(&program, &mut state),
            &mut builder,
            1,
        );
        let reference_columns = builder.snapshot_columns_to_host();
        drop(builder.take_columns());
        println!("run + fused H2D (1 thread): {total:?}");
        let reference_advice = state.advice(&program).to_vec();
        let reference_lookups = state.lookups(&program).to_vec();

        // Warm-tape reruns: timing + a consistency check that column
        // placement is independent of thread-count chunking. (Fresh-tape
        // correctness is covered by `graph_executor_matches_halo2_backend`.)
        for num_threads in [4, 8, 12] {
            let mut builder = fused_builder();
            let total = timed_run(
                &mut GraphExecutor::new(&program, &mut state),
                &mut builder,
                num_threads,
            );
            let columns = builder.snapshot_columns_to_host();
            drop(builder.take_columns());
            println!("run + fused H2D ({num_threads} threads): {total:?}");
            assert_eq!(state.advice(&program), &reference_advice[..]);
            assert_eq!(state.lookups(&program), &reference_lookups[..]);
            assert_eq!(columns.len(), reference_columns.len());
            for (i, (col, reference)) in columns.iter().zip(&reference_columns).enumerate() {
                assert!(
                    col == reference,
                    "device column {i} mismatch vs 1-thread reference ({num_threads} threads)"
                );
            }
        }
    }

    /// Benchmarks the witness-gen pipeline `prove_wrapped` runs before SNARK
    /// generation: `generate_witness` + `FusedColumnBuilder` H2D copies. SNARK
    /// generation itself is excluded.
    ///
    /// Runs the pipeline three times per thread count to surface cold/warm
    /// timings. Run with:
    /// ```text
    /// cargo test --profile fast -p openvm-static-verifier \
    ///     --features evm-prove,halo2-gpu \
    ///     -- --ignored --nocapture \
    ///        graph_executor_prove_wrapped_pipeline
    /// ```
    #[test]
    #[cfg(all(feature = "evm-prove", feature = "halo2-gpu"))]
    #[ignore = "requires CUDA GPU; slow (fixture STARK prove + halo2 keygen)"]
    fn graph_executor_prove_wrapped_pipeline() {
        let (pk, proof) = keygen_fixture_static_verifier();
        let mut state = GraphExecutorState::new(&pk.graph_program);

        // Warm-up pays one-time costs (device init, first column allocation)
        // so the timed loop measures per-proof work only.
        let start = Instant::now();
        let (warmup_advice, _) = pk.generate_witness(&proof, 1, &mut state);
        println!("pipeline warm-up (1 thread): {:?}", start.elapsed());
        drop(warmup_advice);

        for num_threads in [4, 8, 12] {
            for iter in 0..3 {
                let start = Instant::now();
                let (gpu_advice, _instances) = pk.generate_witness(&proof, num_threads, &mut state);
                println!(
                    "pipeline (threads={num_threads}, iter={iter}): {:?}",
                    start.elapsed()
                );
                drop(gpu_advice);
            }
        }
    }
}
