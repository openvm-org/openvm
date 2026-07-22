//! Parallel executor for the graph IR recorded by [`Halo2IRBuilder`].
//!
//! Lowering ([`GraphExecutor::new`]) flattens every graph node into a
//! [`GraphCoreInst`]: ranges into flat operand-offset / constant-skip tables plus
//! the node's precomputed tape offsets and lengths from
//! [`NodeMeta`](crate::halo2_ir_builder::NodeMeta). The witness tape has layout
//! `[advice | lookups | consts]` — fixed-column constants are deduplicated into
//! the trailing region — so operand gathering is uniform: every operand is a
//! tape offset.
//!
//! Execution is split in two phases:
//! 1. **Input population**: `load_proof_wire` streams the proof witnesses into the tape through the
//!    [`PopulateInputs`] impl, which replays the recorded `LoadWitness` instructions in node order.
//! 2. **[`GraphExecutor::run`]**: the remaining instructions, sorted by dataflow level, are claimed
//!    off a shared atomic cursor by worker threads. Synchronization is per-instruction: each worker
//!    spin-waits on its parents' done flags before executing — there are no barriers. Input
//!    instructions are excluded from the schedule; their cells were written during population.

use std::{
    collections::HashMap,
    sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering},
};

use halo2_base::halo2_proofs::{arithmetic::Field as _, halo2curves::bn256::Fr};
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2Config as RootConfig,
    openvm_stark_backend::{
        p3_field::{BasedVectorSpace, PrimeField64},
        proof::Proof,
    },
    p3_baby_bear::BabyBear,
};
#[cfg(feature = "halo2-gpu")]
use halo2_base::halo2_proofs::cuda::utils::HALO2_GPU_CTX;
#[cfg(feature = "halo2-gpu")]
use openvm_cuda_common::d_buffer::DeviceBuffer;

use crate::{
    chip_traits::{ChipBase, PopulateInputs},
    circuit::StaticVerifierCircuit,
    field::baby_bear::{
        BabyBearExt4, BabyBearWire, ReducedBabyBearExt4Wire, ReducedBabyBearWire, BABYBEAR_MAX_BITS,
    },
    halo2_ir_builder::{GraphCell, Halo2IRBuilder, Halo2Opcode},
    halo2_opcode_impl::{interpret_op, UNMATERIALIZED},
    stages::full_pipeline::load_proof_wire,
};

/// One lowered graph node (replayed via
/// [`interpret_op`](crate::halo2_opcode_impl::interpret_op)).
#[derive(Copy, Clone, Debug)]
struct GraphCoreInst {
    opcode: Halo2Opcode,
    /// `operand_offsets`/`operand_bits[lo..hi]`.
    args: (u32, u32),
    /// `const_inds[lo..hi]`: the node's constant-skip indices.
    const_inds: (u32, u32),
    /// Absolute offset into the advice region the node writes at.
    ctx_offset: u32,
    ctx_len: u32,
    /// Offset into the lookup region the node writes at.
    lookup_offset: u32,
    lookups_len: u32,
    /// `dep_inds[lo..hi]`: parent instruction indices.
    dep_list: (u32, u32),
}

/// One compute writer in the tape-emission-order release schedule (see
/// [`GraphExecutor::release_order`]).
#[derive(Copy, Clone)]
struct ReleaseEntry {
    /// Index into `insts`/`flags`.
    inst: u32,
    /// Exclusive end of the advice range this writer releases on completion.
    /// The range starts at the previous entry's `advice_end` (0 for the
    /// first), so input-populated gap cells between consecutive writers are
    /// released together with the following writer.
    advice_end: u32,
    /// Same as `advice_end`, for the range-check (lookup) tape.
    lookup_end: u32,
}

/// Shared mutable witness tape.
#[derive(Clone, Copy)]
struct TapePtr(*mut Fr);
#[allow(unsafe_code)]
unsafe impl Send for TapePtr {}
#[allow(unsafe_code)]
unsafe impl Sync for TapePtr {}

pub struct GraphExecutor {
    advice_cells: usize,
    lookup_cells: usize,
    /// Layout: `[advice_cells | lookup_cells | consts]`.
    tape: Vec<Fr>,
    /// Flattened operand tape offsets for all instructions.
    operand_offsets: Vec<u32>,
    /// Bit bound of each operand, parallel to `operand_offsets`.
    operand_bits: Vec<u16>,
    /// Flattened constant-skip indices for all instructions.
    const_inds: Vec<u32>,
    /// Compute instructions, reordered by dataflow level.
    insts: Vec<GraphCoreInst>,
    /// `LoadWitness` instructions, in node order.
    input_insts: Vec<GraphCoreInst>,
    input_cursor: usize,
    lookup_bits: usize,
    /// Compute writers sorted by tape-emission order (ascending write
    /// offsets). Entry `i` releases `advice[prev.advice_end..advice_end)` /
    /// `lookups[prev.lookup_end..lookup_end)` once its instruction's flag is
    /// set; the ends tile both tapes, with the last entry extended to cover
    /// any trailing input-populated cells.
    release_order: Vec<ReleaseEntry>,
    /// Flattened parent-instruction indices; `insts[i].dep_list` names a slice
    /// of this vector. See [`GraphCoreInst::dep_list`].
    dep_inds: Vec<u32>,
    /// Per-instruction "done" flag. Set to the current `phase`
    /// once the instruction finishes; workers spin on parent flags waiting for
    /// them to reach the current phase before executing.
    flags: Vec<AtomicU8>,
    /// Wrapping run counter; each [`Self::run`] bumps it (skipping 0) and
    /// stamps finished instructions' `flags` with it, so flags never need
    /// resetting between runs.
    phase: u8,
}

impl GraphExecutor {
    pub fn new(ir: Halo2IRBuilder) -> Self {
        let advice_cells = ir.total_ctx_len();
        let lookup_cells = ir.total_lookups_len();
        let lookup_bits = ir.lookup_bits();
        let const_base = advice_cells + lookup_cells;
        let levels = ir.node_levels();

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
                _ => compute.push((levels[node.id as usize], inst)),
            }
        }

        // Emission-order release schedule: `compute` is still in the IR's
        // node-emission order here, where both write offsets are monotone, so
        // the writers' (disjoint) write ranges tile each tape in order. Each
        // writer releases its own range plus the input-populated gap before
        // it. `inst` holds the emission index until the remap below.
        let mut release_order: Vec<ReleaseEntry> = Vec::new();
        let (mut prev_a, mut prev_l) = (0u32, 0u32);
        for (i, (_, inst)) in compute.iter().enumerate() {
            if inst.ctx_len == 0 && inst.lookups_len == 0 {
                continue;
            }
            debug_assert!(inst.ctx_offset >= prev_a && inst.lookup_offset >= prev_l);
            prev_a = inst.ctx_offset + inst.ctx_len;
            prev_l = inst.lookup_offset + inst.lookups_len;
            release_order.push(ReleaseEntry {
                inst: i as u32,
                advice_end: prev_a,
                lookup_end: prev_l,
            });
        }
        if let Some(last) = release_order.last_mut() {
            last.advice_end = advice_cells as u32;
            last.lookup_end = lookup_cells as u32;
        }

        // Stable sort by level keeps node order within a level (write
        // locality); `order[sorted]` is the emission index of the instruction
        // placed at sorted position `sorted`.
        let mut order: Vec<u32> = (0..compute.len() as u32).collect();
        order.sort_by_key(|&i| compute[i as usize].0);
        let mut insts: Vec<GraphCoreInst> = order.iter().map(|&i| compute[i as usize].1).collect();
        let mut emission_to_sorted = vec![0u32; compute.len()];
        for (sorted, &emission) in order.iter().enumerate() {
            emission_to_sorted[emission as usize] = sorted as u32;
        }
        for e in &mut release_order {
            e.inst = emission_to_sorted[e.inst as usize];
        }
        drop(compute);

        let mut tape = vec![Fr::ZERO; const_base + consts.len()];
        tape[const_base..].copy_from_slice(&consts);

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

        let flags: Vec<AtomicU8> = (0..insts.len()).map(|_| AtomicU8::new(0)).collect();

        GraphExecutor {
            advice_cells,
            lookup_cells,
            tape,
            operand_offsets,
            operand_bits,
            const_inds,
            insts,
            input_insts,
            input_cursor: 0,
            lookup_bits,
            release_order,
            dep_inds,
            flags,
            phase: 0,
        }
    }

    /// The advice (context) tape; matches `Context::advice_cells()` of the halo2
    /// backend after [`Self::run`].
    pub fn advice(&self) -> &[Fr] {
        &self.tape[..self.advice_cells]
    }

    /// The range-check tape; matches the values sent to `add_cell_to_lookup`.
    pub fn lookups(&self) -> &[Fr] {
        &self.tape[self.advice_cells..self.advice_cells + self.lookup_cells]
    }

    /// Rewinds the input cursor so a new proof's witnesses can be populated. Every
    /// compute cell is fully overwritten by the next [`Self::run`], so no other state
    /// needs clearing.
    pub fn reset(&mut self) {
        self.input_cursor = 0;
    }

    /// Replays the next recorded input instruction with the proof `value`,
    /// writing its full tape footprint (witness cell plus any range-check cells).
    fn populate_input(&mut self, expected: Halo2Opcode, value: Fr) -> usize {
        debug_assert!(matches!(expected, Halo2Opcode::LoadWitness));
        let inst = *self
            .input_insts
            .get(self.input_cursor)
            .expect("more input loads than recorded input instructions");
        debug_assert_eq!(
            inst.opcode, expected,
            "input load {} kind mismatch",
            self.input_cursor
        );
        self.input_cursor += 1;
        let advice = &mut self.tape[..self.advice_cells];
        advice[inst.ctx_offset as usize] = value;
        debug_assert!(inst.ctx_len == 1);
        debug_assert!(inst.lookups_len == 0);
        inst.ctx_offset as usize
    }

    /// Evaluates the compute schedule with `num_threads` workers, barrier-free:
    /// workers claim instructions off an atomic cursor (level-sorted, so roughly
    /// topological), spin-wait on their parents' done flags, execute, and
    /// Release-store their own flag. Flags are stamped with this run's `phase`
    /// instead of being zeroed between runs; stale values are ignored.
    ///
    /// Meanwhile the calling thread walks [`Self::release_order`] and streams
    /// newly-materialized tape ranges through `on_delta(advice_offset,
    /// advice_delta, lookup_offset, lookup_delta)`: a bounded spin per writer
    /// flag extends the pending contiguous range on success and defers the
    /// writer to a retry pass on timeout, so one slow writer only dams its own
    /// cells, not the prefix behind it. Flushes are batched between
    /// `MIN_FLUSH_CELLS` and `MAX_FLUSH_CELLS`.
    #[allow(unsafe_code)]
    pub fn run<F>(&mut self, num_threads: usize, mut on_delta: F)
    where
        F: FnMut(usize, &[Fr], usize, &[Fr]),
    {
        assert!(num_threads > 0);
        assert_eq!(
            self.input_cursor,
            self.input_insts.len(),
            "all proof inputs must be populated before run"
        );

        // Bump phase. Skip 0 so a freshly-constructed executor (with `flags`
        // all zero) never accidentally reports flags as "ready" for phase 0.
        self.phase = self.phase.wrapping_add(1);
        if self.phase == 0 {
            self.phase = 1;
        }
        let phase = self.phase;

        // Detach the tape so threads share `&self` without aliasing the buffer.
        let mut tape = std::mem::take(&mut self.tape);
        let tape_ptr = TapePtr(tape.as_mut_ptr());
        let advice_cells = self.advice_cells;
        let this = &*self;
        let release_order = self.release_order.as_slice();
        let n_insts = self.insts.len();
        // Workers `fetch_add(1)` to claim instructions; the level-sorted order
        // makes claims roughly topological, keeping parent spin-waits short.
        let claim_cursor = AtomicUsize::new(0);

        // A panicked worker leaves its flag unset forever, which would livelock
        // peer spin-waits and the release walk's retry passes. Workers set this
        // on unwind; every unbounded wait checks it and panics instead.
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
                        let inst = &this.insts[idx];
                        // Spin-wait until every parent's flag matches this
                        // run's phase.
                        let (dep_lo, dep_hi) = inst.dep_list;
                        for d in dep_lo as usize..dep_hi as usize {
                            let parent = this.dep_inds[d] as usize;
                            while this.flags[parent].load(Ordering::Acquire) != phase {
                                assert!(
                                    !poisoned.load(Ordering::Relaxed),
                                    "graph executor worker panicked"
                                );
                                std::hint::spin_loop();
                            }
                        }
                        this.eval_inst(inst, tape_ptr, &mut args, &mut bits);
                        // Release-publishes this instruction's tape writes to
                        // whoever Acquire-loads the flag and observes `phase`.
                        this.flags[idx].store(phase, Ordering::Release);
                    }
                });
            }

            // Release walk on the calling thread (see the `run` doc).
            {
                /// Flag-poll attempts before the walk gives up on a writer,
                /// defers it to the next retry pass, and moves on.
                const MAX_SPIN_TRIES: usize = 32;
                /// Pending ranges smaller than this (advice + lookup cells)
                /// are deferred (as already-ready entries) and re-merged with
                /// neighbours in later passes instead of being flushed as tiny
                /// H2D copies.
                const MIN_FLUSH_CELLS: u32 = 8 * 1024;
                /// Pending ranges flush once they reach this many advice
                /// cells, so a long completed run streams out incrementally
                /// instead of as one giant callback.
                const MAX_FLUSH_CELLS: u32 = 1 << 20;
                /// `inst` marker for a deferred, already-materialized range.
                const READY_SENTINEL: u32 = u32::MAX;

                /// `(a_start, a_end, l_start, l_end)` tape range.
                type Range = (u32, u32, u32, u32);
                /// A `Range` gated on `flags[inst]` (or none, for the
                /// sentinel).
                type Entry = (u32, u32, u32, u32, u32);

                let try_wait = |inst: u32| {
                    if inst == READY_SENTINEL {
                        return true;
                    }
                    let flag = &this.flags[inst as usize];
                    for _ in 0..MAX_SPIN_TRIES {
                        if flag.load(Ordering::Acquire) == phase {
                            return true;
                        }
                        std::hint::spin_loop();
                    }
                    false
                };
                // Fires `on_delta` for the range — unless it is
                // sub-`MIN_FLUSH_CELLS` and `defer_to` is given, in which case
                // it is queued as a ready entry for a later merge.
                //
                // Safety: `try_wait`'s Acquire load pairs with each writer's
                // Release store, ordering the tape writes before this read;
                // the remaining cells were input-populated before `run`.
                // Writer ranges are disjoint, so nothing writes into a
                // released range.
                let mut flush_or_defer = |r: Range, defer_to: Option<&mut Vec<Entry>>| {
                    let (a_start, a_end, l_start, l_end) = r;
                    if a_start == a_end && l_start == l_end {
                        return;
                    }
                    if let Some(defer) = defer_to {
                        if (a_end - a_start) + (l_end - l_start) < MIN_FLUSH_CELLS {
                            defer.push((READY_SENTINEL, a_start, a_end, l_start, l_end));
                            return;
                        }
                    }
                    let advice_delta: &[Fr] = unsafe {
                        std::slice::from_raw_parts(
                            (tape_ptr.0 as *const Fr).add(a_start as usize),
                            (a_end - a_start) as usize,
                        )
                    };
                    let lookup_delta: &[Fr] = unsafe {
                        std::slice::from_raw_parts(
                            (tape_ptr.0 as *const Fr).add(advice_cells + l_start as usize),
                            (l_end - l_start) as usize,
                        )
                    };
                    on_delta(
                        a_start as usize,
                        advice_delta,
                        l_start as usize,
                        lookup_delta,
                    );
                };

                // Pass 0: completed writers extend the pending range (its end
                // is always the next entry's start); a timeout flushes the
                // pending range and queues the writer for retry. Entries are
                // pushed in walk order, so `failed` stays emission-ordered.
                let mut failed: Vec<Entry> = Vec::new();
                let mut pend: Range = (0, 0, 0, 0);
                for &ReleaseEntry {
                    inst,
                    advice_end,
                    lookup_end,
                } in release_order
                {
                    if try_wait(inst) {
                        pend.1 = advice_end;
                        pend.3 = lookup_end;
                        if pend.1 - pend.0 >= MAX_FLUSH_CELLS {
                            flush_or_defer(pend, None);
                            pend = (advice_end, advice_end, lookup_end, lookup_end);
                        }
                    } else {
                        flush_or_defer(pend, Some(&mut failed));
                        failed.push((inst, pend.1, advice_end, pend.3, lookup_end));
                        pend = (advice_end, advice_end, lookup_end, lookup_end);
                    }
                }
                if failed.is_empty() {
                    flush_or_defer(pend, None);
                } else {
                    flush_or_defer(pend, Some(&mut failed));
                }

                // Retry passes: the same walk over the leftovers, merging
                // adjacent ranges. Deferral is only allowed while flag-gated
                // entries remain — once all are ready nothing new can merge,
                // so the final pass flushes everything (ensuring termination).
                while !failed.is_empty() {
                    assert!(
                        !poisoned.load(Ordering::Relaxed),
                        "graph executor worker panicked"
                    );
                    let allow_defer = failed.iter().any(|&(inst, ..)| inst != READY_SENTINEL);
                    let mut still: Vec<Entry> = Vec::new();
                    let mut pend: Option<Range> = None;
                    for &(inst, a_start, a_end, l_start, l_end) in &failed {
                        if try_wait(inst) {
                            let merged = match pend {
                                Some(p) if p.1 == a_start && p.3 == l_start => {
                                    (p.0, a_end, p.2, l_end)
                                }
                                Some(p) => {
                                    flush_or_defer(
                                        p,
                                        if allow_defer { Some(&mut still) } else { None },
                                    );
                                    (a_start, a_end, l_start, l_end)
                                }
                                None => (a_start, a_end, l_start, l_end),
                            };
                            if merged.1 - merged.0 >= MAX_FLUSH_CELLS {
                                flush_or_defer(merged, None);
                                pend = None;
                            } else {
                                pend = Some(merged);
                            }
                        } else {
                            if let Some(p) = pend.take() {
                                flush_or_defer(p, Some(&mut still));
                            }
                            still.push((inst, a_start, a_end, l_start, l_end));
                        }
                    }
                    if let Some(p) = pend {
                        if allow_defer && !still.is_empty() {
                            flush_or_defer(p, Some(&mut still));
                        } else {
                            flush_or_defer(p, None);
                        }
                    }
                    failed = still;
                }
            }
        });

        self.tape = tape;
    }

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
            // Safety: operand cells are input/const (prefilled before `run`)
            // or parent outputs, published by the parent-flag Acquire in `run`
            // before this instruction executes.
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

impl ChipBase for GraphExecutor {
    /// Wires are absolute offsets into the executor's tape.
    type F = usize;
}

impl PopulateInputs for GraphExecutor {
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

/// Reusable graph-based witness generator for a [`StaticVerifierCircuit`].
///
/// Records the full populate trace (STARK constraints + DAG onion-commit pin +
/// public-value extraction) once, then regenerates the witness for any proof of the same
/// static shape via [`Self::witness_gen`].
pub struct GraphProver {
    executor: GraphExecutor,
    /// Advice-tape offsets of the circuit's public values, in instance order.
    pv_offsets: Vec<usize>,
}

impl GraphProver {
    /// Records `circuit`'s populate trace into a graph IR and lowers it. Any valid proof
    /// for the circuit's static shape works as the `representative_proof`.
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
        Self {
            executor: GraphExecutor::new(ir),
            pv_offsets,
        }
    }

    /// Populates `proof`'s witnesses, evaluates the graph with `num_threads` compute
    /// threads, and streams newly-materialized advice/lookup tape ranges (offset +
    /// delta slice) through `on_delta` (see [`GraphExecutor::run`] for the closure
    /// contract). Returns the circuit's public values; any output from the callback
    /// itself must be plumbed out via shared state captured by the closure.
    pub fn witness_gen<F>(
        &mut self,
        circuit: &StaticVerifierCircuit,
        proof: &Proof<RootConfig>,
        num_threads: usize,
        on_delta: F,
    ) -> Vec<Fr>
    where
        F: FnMut(usize, &[Fr], usize, &[Fr]),
    {
        self.executor.reset();
        tracing::info_span!("populate_inputs").in_scope(|| {
            load_proof_wire(&mut self.executor, proof, &circuit.log_heights_per_air);
        });
        tracing::info_span!("executor_run", num_threads)
            .in_scope(|| self.executor.run(num_threads, on_delta));
        tracing::info_span!("collect_pvs").in_scope(|| {
            let advice = self.executor.advice();
            self.pv_offsets
                .iter()
                .map(|&offset| advice[offset])
                .collect()
        })
    }

    /// Total number of advice-tape cells written per [`Self::witness_gen`].
    pub fn total_advice_cells(&self) -> usize {
        self.executor.advice().len()
    }

    /// Total number of range-check tape cells written per [`Self::witness_gen`].
    pub fn total_lookup_cells(&self) -> usize {
        self.executor.lookups().len()
    }

    /// The advice (context) tape of the last [`Self::witness_gen`].
    pub fn advice(&self) -> &[Fr] {
        self.executor.advice()
    }

    /// The range-check tape of the last [`Self::witness_gen`].
    pub fn lookups(&self) -> &[Fr] {
        self.executor.lookups()
    }
}

/// Streams graph-executor tape deltas straight into device-resident advice columns.
///
/// Placement is a pure function of the tape offset, so disjoint deltas may arrive
/// in any order. The layout mirrors `PagedWitnessContext::push_advice` (gate
/// columns split at pinned break points, duplicating the break-row value at row 0
/// of the next column so the gate-overlap copy constraint holds) and
/// `BaseCircuitBuilder::assign_lookups_in_phase` (lookup columns fill round-robin:
/// value `i` at column `i % L`, row `i / L`).
///
/// Columns are zero-filled `DeviceBuffer`s allocated on the first [`Self::append`];
/// each contiguous delta segment is copied H2D directly into its row range, so no
/// host-side column buffer is ever materialized.
#[cfg(feature = "halo2-gpu")]
pub struct FusedColumnBuilder {
    // ---- Config (immutable after `new`) ------------------------------------
    n: usize,
    num_advice_columns: usize,
    /// Pinned break points, in order (`break_points[c]` is gate column `c`'s
    /// break row).
    break_points: Vec<usize>,
    /// Absolute advice-tape offset of row 0 of each gate column. Row 0
    /// duplicates the previous column's break-point value, so
    /// `col_starts[c + 1] = col_starts[c] + break_points[c]`.
    col_starts: Vec<usize>,
    /// Physical column indices of the range-check lookup advice columns.
    lookup_col_indices: Vec<usize>,

    // ---- Device columns (lazily allocated on first `append`) ---------------
    device_columns: Vec<DeviceBuffer<Fr>>,
}

#[cfg(feature = "halo2-gpu")]
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
            device_columns: Vec::new(),
        }
    }

    fn ensure_allocated(&mut self) {
        if !self.device_columns.is_empty() {
            return;
        }
        self.device_columns.reserve_exact(self.num_advice_columns);
        for _ in 0..self.num_advice_columns {
            let buf: DeviceBuffer<Fr> =
                DeviceBuffer::<Fr>::with_capacity_on(self.n, &HALO2_GPU_CTX);
            buf.fill_zero_on(&HALO2_GPU_CTX)
                .expect("zero-fill advice column");
            self.device_columns.push(buf);
        }
    }

    /// Copies `advice_delta` (advice-tape range starting at absolute offset
    /// `advice_offset`) and `lookup_delta` (range-check-tape range starting at
    /// `lookup_offset`) into the device columns. Placement is a pure function
    /// of the offsets, so disjoint deltas may arrive in any order.
    pub fn append(
        &mut self,
        advice_offset: usize,
        advice_delta: &[Fr],
        lookup_offset: usize,
        lookup_delta: &[Fr],
    ) {
        self.ensure_allocated();

        // --- Gate stream: contiguous H2D per (column, row-range) segment ----
        //
        // Gate column `c` covers tape offsets `[col_starts[c], col_starts[c + 1]]`
        // inclusive at both ends: the shared endpoint is the break value,
        // duplicated at `(c, break_points[c])` and `(c + 1, 0)`. On crossing a
        // break, `delta_pos -= 1` rewinds so the break value is re-emitted as
        // row 0 of the next column; a delta starting exactly on a column start
        // likewise begins at the earlier placement so the duplicate is written.
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
                self.device_columns[col]
                    .mut_slice(row..row + take)
                    .copy_from_host(src, &HALO2_GPU_CTX)
                    .expect("H2D advice gate segment");
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

        // --- Lookup stream: one gathered H2D per lookup column per call ------
        //
        // Value `i` lands at column `L[i % L]`, row `i / L`: a delta's values
        // for one column are strided in delta space but row-contiguous, so
        // gather each stride into a host buffer and issue one H2D per column.
        let l = self.lookup_col_indices.len();
        let k = lookup_offset;
        let n_l = lookup_delta.len();
        if n_l > 0 {
            for c in 0..l {
                let start_j = (c + l - k % l) % l;
                if start_j >= n_l {
                    continue;
                }
                let n_values = (n_l - start_j).div_ceil(l);
                let start_row = (k + start_j) / l;
                let host_buf: Vec<Fr> = (0..n_values)
                    .map(|i| lookup_delta[start_j + i * l])
                    .collect();
                self.device_columns[self.lookup_col_indices[c]]
                    .mut_slice(start_row..start_row + n_values)
                    .copy_from_host(&host_buf, &HALO2_GPU_CTX)
                    .expect("H2D lookup column gather");
            }
        }
    }

    /// Consumes the device columns, leaving the builder empty.
    pub fn take_device_columns(&mut self) -> Vec<DeviceBuffer<Fr>> {
        assert!(
            !self.device_columns.is_empty(),
            "take_device_columns: no data was ever appended",
        );
        std::mem::take(&mut self.device_columns)
    }

    /// Diagnostic-only: D2H each device column back into host `Vec<Fr>`s so the
    /// caller can byte-compare against the legacy `BaseCircuitBuilder` +
    /// `synthesize_witness_shplonk` path. Not used on the hot prove path.
    pub fn snapshot_columns_to_host(&self) -> Vec<Vec<Fr>> {
        use openvm_cuda_common::copy::MemCopyD2H;
        self.device_columns
            .iter()
            .map(|d| d.to_host_on(&HALO2_GPU_CTX).expect("D2H advice column"))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::Arc,
        time::{Duration, Instant},
    };

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
    use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

    use super::*;
    use crate::{
        halo2_backend::Halo2Backend,
        stages::{full_pipeline::load_proof_wire, proof_shape::log_heights_per_air_from_proof},
        test_fixtures::{fixture_circuit_and_proof, FIXTURE_K},
        Halo2Params, StaticVerifierCircuit, StaticVerifierProvingKey, StaticVerifierShape,
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
    ) -> GraphExecutor {
        let mut ir = Halo2IRBuilder::new(LOOKUP_BITS);
        circuit.populate_verify_stark_constraints(&mut ir, proof);
        let mut executor = GraphExecutor::new(ir);
        load_proof_wire(&mut executor, proof, log_heights_per_air);
        executor.run(num_threads, |_, _, _, _| ());
        executor
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

        let executor = build_and_run(&circuit, &proof, &log_heights_per_air, 4);
        assert_eq!(executor.advice().len(), real_advice.len(), "advice len");
        assert_eq!(executor.advice(), &real_advice[..], "advice tape");
        assert_eq!(executor.lookups(), &real_lookups[..], "range tape");

        // Determinism across schedules: single-threaded run matches.
        let sequential = build_and_run(&circuit, &proof, &log_heights_per_air, 1);
        assert_eq!(sequential.advice(), executor.advice());
        assert_eq!(sequential.lookups(), executor.lookups());
    }

    /// Runs the executor, streaming each delta through `builder`; returns the
    /// wall time of the combined run + fused H2D copies.
    fn timed_run(
        executor: &mut GraphExecutor,
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

    /// Production-path setup with no cached artifacts: STARK-proves the shared
    /// root-shaped fixture, then runs [`StaticVerifierProvingKey::keygen`] on an
    /// in-memory SRS to get a real pinning (config params, break points, halo2 pk).
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
    #[ignore = "requires CUDA GPU; slow (fixture STARK prove + halo2 keygen)"]
    fn graph_executor_root_proof() {
        use halo2_base::{
            gates::circuit::MaybeRangeConfig,
            halo2_proofs::{halo2curves::bn256::G1Affine, plonk::create_constraint_system},
        };

        let (pk, proof) = keygen_fixture_static_verifier();
        let metadata = &pk.pinning.metadata;
        let log_heights_per_air = log_heights_per_air_from_proof(&proof);

        // Physical column layout for the FusedColumnBuilder (mirrors
        // `StaticVerifierProvingKey::run_witness_gen_pipeline`, minus the pk).
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
        let mut executor = GraphExecutor::new(ir);
        println!(
            "lowering: {:?} ({} insts)",
            start.elapsed(),
            executor.insts.len()
        );

        let start = Instant::now();
        load_proof_wire(&mut executor, &proof, &log_heights_per_air);
        println!("input population: {:?}", start.elapsed());

        let mut builder = fused_builder();
        let total = timed_run(&mut executor, &mut builder, 1);
        let reference_columns = builder.snapshot_columns_to_host();
        drop(builder.take_device_columns());
        println!("run + fused H2D (1 thread): {total:?}");
        let reference_advice = executor.advice().to_vec();
        let reference_lookups = executor.lookups().to_vec();

        // Reruns reuse the warm tape (fresh-tape correctness vs the real backend
        // is covered by `graph_executor_matches_halo2_backend`); these are timing
        // plus consistency checks: thread count changes the delta chunking, so
        // identical device columns show placement is chunking-independent.
        for num_threads in [4, 8, 12] {
            let mut builder = fused_builder();
            let total = timed_run(&mut executor, &mut builder, num_threads);
            let columns = builder.snapshot_columns_to_host();
            drop(builder.take_device_columns());
            println!("run + fused H2D ({num_threads} threads): {total:?}");
            assert_eq!(executor.advice(), &reference_advice[..]);
            assert_eq!(executor.lookups(), &reference_lookups[..]);
            assert_eq!(columns.len(), reference_columns.len());
            for (i, (col, reference)) in columns.iter().zip(&reference_columns).enumerate() {
                assert!(
                    col == reference,
                    "device column {i} mismatch vs 1-thread reference ({num_threads} threads)"
                );
            }
        }
    }

    /// Synthetic benchmark of the full witness-generation pipeline that
    /// [`StaticVerifierProvingKey::prove_wrapped`] runs before handing off to
    /// `snark_verifier_sdk`: `GraphProver::witness_gen` (parallel graph-IR
    /// evaluation) + the [`FusedColumnBuilder`] streaming per-column H2D
    /// copies. Stops after `builder.take_device_columns()` — SNARK generation
    /// itself is excluded.
    ///
    /// Builds the proving key and root proof from [`RootShapedFixture`] (see
    /// [`keygen_fixture_static_verifier`]). Runs the pipeline three times per
    /// thread count so the first (cold) and subsequent (warm) numbers are
    /// visible.
    ///
    /// Run with:
    /// ```text
    /// cargo test --profile fast -p openvm-static-verifier \
    ///     --features evm-prove,halo2-gpu \
    ///     -- --ignored --nocapture \
    ///        graph_executor_prove_wrapped_pipeline
    /// ```
    #[test]
    #[cfg(feature = "evm-prove")]
    #[ignore = "requires CUDA GPU; slow (fixture STARK prove + halo2 keygen)"]
    fn graph_executor_prove_wrapped_pipeline() {
        let (pk, proof) = keygen_fixture_static_verifier();

        // The GraphProver's IR tape is built eagerly at keygen; this warm-up
        // pays the remaining one-time costs (device init, first fused-column
        // allocation) so the timed loop measures only per-proof work.
        let start = Instant::now();
        let (warmup_advice, _) = pk.run_witness_gen_pipeline(&proof, 1, None);
        println!("pipeline warm-up (1 thread): {:?}", start.elapsed());
        drop(warmup_advice);

        for num_threads in [4, 8, 12] {
            for iter in 0..3 {
                let start = Instant::now();
                let (gpu_advice, _instances) =
                    pk.run_witness_gen_pipeline(&proof, num_threads, None);
                println!(
                    "pipeline (threads={num_threads}, iter={iter}): {:?}",
                    start.elapsed()
                );
                drop(gpu_advice);
            }
        }
    }
}
