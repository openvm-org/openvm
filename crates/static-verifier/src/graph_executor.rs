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
//!    [`PopulateInputs`] impl, which replays the recorded `LoadWitness` / `LoadBBReducedWitness`
//!    instructions in node order.
//! 2. **[`GraphExecutor::run`]**: the remaining instructions, reordered by dataflow level, execute
//!    level by level. Instructions on the same level are independent, so threads claim chunks of a
//!    level off an atomic cursor; a barrier separates levels so operands are always fully written
//!    before they are read. Input instructions are excluded from the schedule (no-ops at run time —
//!    their cells were written during population).

// `Instant` used only by the (currently commented-out) per-thread profiling
// timers in `GraphExecutor::run`; kept imported so uncommenting them Just Works.
#[allow(unused_imports)]
use std::time::Instant;
use std::{
    collections::HashMap,
    sync::atomic::{AtomicU8, AtomicUsize, Ordering},
    time::Duration,
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
    /// `dep_inds[lo..hi]`: indices (into `insts`) of the compute instructions
    /// whose output cells this instruction reads as operands. Input
    /// instructions are excluded — they run before [`GraphExecutor::run`]
    /// starts, so their flags are always considered "ready". Parents always
    /// have strictly lower index than this instruction (insts are level-sorted
    /// and dependencies point to strictly-lower levels).
    dep_list: (u32, u32),
}

/// Shared mutable witness tape. Sound because every slot is written by exactly
/// one instruction (disjoint precomputed ranges) and reads happen on a later
/// level, ordered after the write by the level barrier.
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
    /// Level `l` is `insts[level_starts[l]..level_starts[l + 1]]`.
    level_starts: Vec<u32>,
    /// `LoadWitness`/`LoadBBReducedWitness` instructions, in node order.
    input_insts: Vec<GraphCoreInst>,
    input_cursor: usize,
    lookup_bits: usize,
    /// After compute level `l` completes, `advice()[0..advice_break_points[l])` is
    /// fully materialized: no future level or thread will write into that range.
    /// Monotonically non-decreasing; the last entry equals `advice_cells`.
    advice_break_points: Vec<usize>,
    /// Same as `advice_break_points`, but for the range-check (lookup) tape.
    lookup_break_points: Vec<usize>,
    /// Flattened parent-instruction indices; `insts[i].dep_list` names a slice
    /// of this vector. See [`GraphCoreInst::dep_list`].
    dep_inds: Vec<u32>,
    /// Per-instruction "done" flag. Set to the current `phase` (see below)
    /// once the instruction finishes; workers spin on parent flags waiting for
    /// them to reach the current phase before executing.
    ///
    /// Comparing against `phase` (rather than a `bool`) avoids having to reset
    /// the buffer between runs — old runs left stale non-current values and
    /// waiters ignore them.
    ///
    /// Unpadded 1-byte flags: the whole buffer is `n_insts` bytes (~500 KB for
    /// the root proof), small enough to stay hot in L2. Adjacent flags share
    /// cache lines, so worker writes can invalidate peers' spin-reads on the
    /// same line — but with only ~1 flag write per line per run on average
    /// (500k writes ÷ 8k lines ≈ 60 writes per line total, spread across the
    /// entire run) the invalidation traffic is bounded and the L2 residency
    /// win dominates.
    flags: Vec<AtomicU8>,
    /// Monotonically-incrementing phase counter, wrapped to `u8`. Each
    /// [`Self::run`] bumps it and skips 0 (so an all-zeros `flags` buffer at
    /// construction time appears "unfinished" to the first run). Wraps every
    /// 255 runs (safe: every flag transitions from the previous phase to the
    /// current one during a run, and 0 is never a valid phase value, so
    /// wrap-around never confuses a stale flag with a current-phase one).
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
                Halo2Opcode::LoadWitness | Halo2Opcode::LoadBBReducedWitness => {
                    input_insts.push(inst)
                }
                _ => compute.push((levels[node.id as usize], inst)),
            }
        }

        // Stable sort keeps node order within a level (write locality).
        compute.sort_by_key(|&(level, _)| level);
        let mut insts = Vec::with_capacity(compute.len());
        let mut level_starts = vec![0u32];
        for (level, inst) in compute {
            while level_starts.len() <= level as usize {
                level_starts.push(insts.len() as u32);
            }
            insts.push(inst);
        }
        level_starts.push(insts.len() as u32);

        let mut tape = vec![Fr::ZERO; const_base + consts.len()];
        tape[const_base..].copy_from_slice(&consts);

        let n_levels = level_starts.len() - 1;
        let (advice_break_points, lookup_break_points) = compute_break_points(
            advice_cells,
            lookup_cells,
            &input_insts,
            &insts,
            &level_starts,
            n_levels,
        );

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
            for offset in ctx_lo..ctx_hi {
                cell_to_compute_inst[offset] = idx as i32;
            }
        }
        let mut dep_inds: Vec<u32> = Vec::new();
        let mut local_deps: Vec<u32> = Vec::new();
        for i in 0..insts.len() {
            local_deps.clear();
            let (arg_lo, arg_hi) = insts[i].args;
            for a in arg_lo as usize..arg_hi as usize {
                let offset = operand_offsets[a] as usize;
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
            insts[i].dep_list = (dep_lo, dep_hi);
        }
        drop(cell_to_compute_inst);

        // Per-instruction ready flags; initialized to 0 so the first run's
        // phase (which starts at 1) sees them all as "not ready" for the
        // current run.
        let flags: Vec<AtomicU8> = (0..insts.len()).map(|_| AtomicU8::new(0)).collect();

        GraphExecutor {
            advice_cells,
            lookup_cells,
            tape,
            operand_offsets,
            operand_bits,
            const_inds,
            insts,
            level_starts,
            input_insts,
            input_cursor: 0,
            lookup_bits,
            advice_break_points,
            lookup_break_points,
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

    /// Number of dataflow levels in the compute schedule.
    pub fn num_levels(&self) -> usize {
        self.level_starts.len() - 1
    }

    /// Number of compute instructions in each level, in level order. Length
    /// equals [`Self::num_levels`]. Excludes input instructions (which run
    /// during population, not during `run`).
    pub fn level_widths(&self) -> Vec<usize> {
        self.level_starts
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .collect()
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
        let inst = *self
            .input_insts
            .get(self.input_cursor)
            .expect("more input loads than recorded input instructions");
        assert_eq!(
            inst.opcode, expected,
            "input load {} kind mismatch",
            self.input_cursor
        );
        self.input_cursor += 1;
        let (advice, rest) = self.tape.split_at_mut(self.advice_cells);
        let ctx = &mut advice[inst.ctx_offset as usize..][..inst.ctx_len as usize];
        let lookups = &mut rest[inst.lookup_offset as usize..][..inst.lookups_len as usize];
        let (lo, hi) = inst.const_inds;
        interpret_op(
            &inst.opcode,
            &[value],
            &[],
            ctx,
            lookups,
            self.lookup_bits,
            &self.const_inds[lo as usize..hi as usize],
        );
        inst.ctx_offset as usize
    }

    /// Evaluates the compute schedule with `num_threads` compute threads using
    /// barrier-free, per-instruction dataflow synchronization: each worker
    /// spin-waits on its parent instructions' ready flags, executes, and flips
    /// its own flag. There is no per-level rendezvous — a thread can run ahead
    /// into later levels as long as its dependencies are satisfied.
    ///
    /// The flag buffer is not zeroed between runs; instead each `run` bumps a
    /// monotonically increasing `phase` counter and threads compare flags
    /// against it. Stale values from previous runs are automatically ignored.
    ///
    /// After every level completes, `on_level_complete(advice_delta, lookup_delta)`
    /// is invoked on the callback thread with the newly-materialized slices —
    /// the callback advances one flag at a time (spin-waiting on each) and
    /// cascades through however many levels are now covered, batching
    /// contiguous completed levels into a single callback call.
    #[allow(unsafe_code)]
    pub fn run<F>(&mut self, num_threads: usize, mut on_level_complete: F)
    where
        F: FnMut(&[Fr], &[Fr]) + Send,
    {
        assert!(num_threads > 0);
        assert_eq!(
            self.input_cursor,
            self.input_insts.len(),
            "all proof inputs must be populated before run"
        );
        let n_levels = self.num_levels();
        assert!(n_levels > 0, "run() called with no compute levels");

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
        let advice_break_points = self.advice_break_points.as_slice();
        let lookup_break_points = self.lookup_break_points.as_slice();
        let n_insts = self.insts.len();
        // Global claim cursor: each worker `fetch_add(1)`s to grab the next
        // instruction index. Since `insts` is level-sorted, this hands them
        // out in (roughly) topological order — workers waiting on a parent
        // typically wait for a peer that already claimed a slightly earlier
        // index. Chunks of 1 keep the contention pattern simple; the fine-
        // grained per-flag spin-wait absorbs the load-balancing role that
        // per-level chunks used to play.
        let claim_cursor = AtomicUsize::new(0);

        std::thread::scope(|s| {
            for _ in 0..num_threads {
                let claim_cursor = &claim_cursor;
                s.spawn(move || {
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
                                std::hint::spin_loop();
                            }
                        }
                        this.eval_inst(inst, tape_ptr, &mut args, &mut bits);
                        // Release-publish this instruction's tape writes to
                        // any downstream instruction (worker or callback) that
                        // Acquire-loads this flag and observes `phase`.
                        this.flags[idx].store(phase, Ordering::Release);
                    }
                });
            }

            // Callback thread: walks flags in inst-index order (which equals
            // level order), waiting for each in turn. Whenever the cursor
            // crosses one or more `level_starts` boundaries, those levels are
            // fully complete and their break-point deltas can be fired.
            //
            // Force whole-value capture of `tape_ptr` (a `TapePtr` — `Send`)
            // since Rust 2021 disjoint captures would otherwise pick up the
            // `!Send` raw pointer field.
            s.spawn(move || {
                let tape_ptr = tape_ptr;
                let mut cursor: usize = 0;
                let mut processed_levels: usize = 0;
                let mut prev_advice_bp = 0usize;
                let mut prev_lookup_bp = 0usize;
                while processed_levels < n_levels {
                    // Wait for the next instruction to finish.
                    while this.flags[cursor].load(Ordering::Acquire) != phase {
                        std::hint::spin_loop();
                    }
                    cursor += 1;
                    // Advance across any levels the cursor has now crossed.
                    let mut newly_complete = processed_levels;
                    while newly_complete < n_levels
                        && this.level_starts[newly_complete + 1] as usize <= cursor
                    {
                        newly_complete += 1;
                    }
                    if newly_complete > processed_levels {
                        let last_l = newly_complete - 1;
                        let curr_advice_bp = advice_break_points[last_l];
                        let curr_lookup_bp = lookup_break_points[last_l];
                        let advice_len = curr_advice_bp - prev_advice_bp;
                        let lookup_len = curr_lookup_bp - prev_lookup_bp;
                        // Safety: every instruction with index in
                        // `[level_starts[processed_levels], level_starts[newly_complete])`
                        // has Release-stored its flag, and the acquire load above
                        // is ordered after those stores. Later-level insts only
                        // write cells at offsets >= curr_advice_bp / curr_lookup_bp
                        // by construction of the break points.
                        let advice_delta: &[Fr] = unsafe {
                            std::slice::from_raw_parts(
                                (tape_ptr.0 as *const Fr).add(prev_advice_bp),
                                advice_len,
                            )
                        };
                        let lookup_delta: &[Fr] = unsafe {
                            std::slice::from_raw_parts(
                                (tape_ptr.0 as *const Fr).add(advice_cells + prev_lookup_bp),
                                lookup_len,
                            )
                        };
                        on_level_complete(advice_delta, lookup_delta);
                        prev_advice_bp = curr_advice_bp;
                        prev_lookup_bp = curr_lookup_bp;
                        processed_levels = newly_complete;
                    }
                }
            });
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
            // Safety: operands are outputs of strictly lower levels (or
            // prefilled input/const cells), written before this level's barrier.
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

/// Computes per-level break points: after level `l`, `advice[0..advice_bp[l])` and
/// `lookup[0..lookup_bp[l])` are fully materialized (no future write lands there).
///
/// Each tape cell is written by exactly one instruction (input or compute). We first
/// label every cell with the level at which it becomes final — `-1` for input-populated
/// cells, `0..n_levels` for compute cells — then, since break points are monotonic in
/// level, we walk both tapes with a single cursor that advances past every cell whose
/// write level is `<= l`.
fn compute_break_points(
    advice_cells: usize,
    lookup_cells: usize,
    input_insts: &[GraphCoreInst],
    insts: &[GraphCoreInst],
    level_starts: &[u32],
    n_levels: usize,
) -> (Vec<usize>, Vec<usize>) {
    const UNWRITTEN: i32 = i32::MAX;
    const INPUT_LEVEL: i32 = -1;
    let mut advice_write_level = vec![UNWRITTEN; advice_cells];
    let mut lookup_write_level = vec![UNWRITTEN; lookup_cells];
    let mark = |cells: &mut [i32], off: u32, len: u32, level: i32| {
        let lo = off as usize;
        let hi = lo + len as usize;
        for slot in &mut cells[lo..hi] {
            *slot = level;
        }
    };
    for inst in input_insts {
        mark(
            &mut advice_write_level,
            inst.ctx_offset,
            inst.ctx_len,
            INPUT_LEVEL,
        );
        mark(
            &mut lookup_write_level,
            inst.lookup_offset,
            inst.lookups_len,
            INPUT_LEVEL,
        );
    }
    for level in 0..n_levels {
        let start = level_starts[level] as usize;
        let end = level_starts[level + 1] as usize;
        for inst in &insts[start..end] {
            mark(
                &mut advice_write_level,
                inst.ctx_offset,
                inst.ctx_len,
                level as i32,
            );
            mark(
                &mut lookup_write_level,
                inst.lookup_offset,
                inst.lookups_len,
                level as i32,
            );
        }
    }
    let scan = |write_level: &[i32], cells: usize| {
        let mut break_points = Vec::with_capacity(n_levels);
        let mut cursor = 0;
        for l in 0..n_levels {
            while cursor < cells && write_level[cursor] <= l as i32 {
                cursor += 1;
            }
            break_points.push(cursor);
        }
        assert_eq!(
            break_points.last().copied().unwrap_or(0),
            cells,
            "final break point must cover the whole tape (all cells written)",
        );
        break_points
    };
    let advice_bp = scan(&advice_write_level, advice_cells);
    let lookup_bp = scan(&lookup_write_level, lookup_cells);
    (advice_bp, lookup_bp)
}

/// Per-worker profiling data collected by [`GraphExecutor::run`]. Currently
/// disabled — see the commented-out timers in `run` — but kept in the tree so
/// re-enabling profiling is a one-line change.
#[allow(dead_code)]
struct WorkerStats {
    /// Number of compute instructions this worker interpreted.
    node_count: usize,
    /// Time spent in `barrier.wait()` across all levels.
    idle_time: Duration,
    /// Nanoseconds spent doing work in each level (excludes barrier wait);
    /// `per_level_work_ns.len() == n_levels`.
    per_level_work_ns: Vec<u64>,
}

/// Logs global per-thread node counts / idle times AND intra-level work-time
/// variance across threads at INFO level via `tracing`.
///
/// Intra-level statistics capture how balanced the per-level work split was:
/// even if aggregate node counts are uniform across threads, individual levels
/// can still be lumpy — the fastest thread then waits at the barrier. Reporting
/// mean/median/max of the per-level standard deviation (across threads) of
/// work time surfaces that imbalance.
#[allow(dead_code)]
fn log_worker_stats(stats: &[WorkerStats]) {
    let n = stats.len();
    if n == 0 {
        return;
    }
    let n_levels = stats[0].per_level_work_ns.len();
    debug_assert!(
        stats.iter().all(|s| s.per_level_work_ns.len() == n_levels),
        "workers disagree on level count"
    );
    let n_f = n as f64;

    // ---- Global per-thread node counts ----------------------------------
    let counts: Vec<usize> = stats.iter().map(|s| s.node_count).collect();
    let total_nodes: usize = counts.iter().sum();
    let mean_nodes = total_nodes as f64 / n_f;
    let variance = counts
        .iter()
        .map(|&c| (c as f64 - mean_nodes).powi(2))
        .sum::<f64>()
        / n_f;
    let std_nodes = variance.sqrt();
    let mut sorted = counts.clone();
    sorted.sort();
    let median_nodes = if n % 2 == 1 {
        sorted[n / 2] as f64
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) as f64 / 2.0
    };

    // ---- Global idle time ------------------------------------------------
    let total_idle: Duration = stats.iter().map(|s| s.idle_time).sum();
    let avg_idle_ms = total_idle.as_secs_f64() * 1000.0 / n_f;
    let total_idle_ms = total_idle.as_secs_f64() * 1000.0;

    // ---- Intra-level work-time std across threads (ns → μs) -------------
    let mut per_level_std_us: Vec<f64> = Vec::with_capacity(n_levels);
    for level in 0..n_levels {
        let sum: f64 = stats
            .iter()
            .map(|s| s.per_level_work_ns[level] as f64)
            .sum();
        let mean = sum / n_f;
        let var: f64 = stats
            .iter()
            .map(|s| (s.per_level_work_ns[level] as f64 - mean).powi(2))
            .sum::<f64>()
            / n_f;
        per_level_std_us.push(var.sqrt() / 1000.0);
    }
    let mean_lvl_std = per_level_std_us.iter().sum::<f64>() / per_level_std_us.len() as f64;
    let max_lvl_std = per_level_std_us
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut sorted_std = per_level_std_us.clone();
    sorted_std.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_lvl_std = if n_levels % 2 == 1 {
        sorted_std[n_levels / 2]
    } else {
        (sorted_std[n_levels / 2 - 1] + sorted_std[n_levels / 2]) / 2.0
    };

    tracing::info!(
        num_threads = n,
        n_levels,
        total_nodes,
        mean_nodes = mean_nodes as u64,
        median_nodes = median_nodes as u64,
        std_nodes = std_nodes as u64,
        avg_idle_ms = format!("{avg_idle_ms:.2}"),
        total_idle_ms = format!("{total_idle_ms:.2}"),
        mean_per_level_work_std_us = format!("{mean_lvl_std:.2}"),
        median_per_level_work_std_us = format!("{median_lvl_std:.2}"),
        max_per_level_work_std_us = format!("{max_lvl_std:.2}"),
        "graph_executor worker stats"
    );
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
        let offset = self.populate_input(
            Halo2Opcode::LoadBBReducedWitness,
            Fr::from(value.as_canonical_u64()),
        );
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
    /// threads, and streams each level's newly-materialized advice/lookup deltas
    /// through `on_level_complete` (see [`GraphExecutor::run`] for the closure
    /// contract). Returns the circuit's public values; any output from the callback
    /// itself must be plumbed out via shared state captured by the closure.
    pub fn witness_gen<F>(
        &mut self,
        circuit: &StaticVerifierCircuit,
        proof: &Proof<RootConfig>,
        num_threads: usize,
        on_level_complete: F,
    ) -> Vec<Fr>
    where
        F: FnMut(&[Fr], &[Fr]) + Send,
    {
        self.executor.reset();
        tracing::info_span!("populate_inputs").in_scope(|| {
            load_proof_wire(&mut self.executor, proof, &circuit.log_heights_per_air);
        });
        tracing::info_span!("executor_run", num_threads)
            .in_scope(|| self.executor.run(num_threads, on_level_complete));
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

    /// Number of compute instructions per level in the underlying executor's
    /// schedule. See [`GraphExecutor::level_widths`].
    pub fn level_widths(&self) -> Vec<usize> {
        self.executor.level_widths()
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

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

    use super::*;
    use crate::{
        halo2_backend::Halo2Backend,
        stages::{full_pipeline::load_proof_wire, proof_shape::log_heights_per_air_from_proof},
        StaticVerifierCircuit,
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
        executor.run(num_threads, |_, _| ());
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
            .advice_cells()
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

    /// Prints summary stats (min/max/mean/median + percentiles) and a
    /// log2-bucketed histogram of a per-level width distribution.
    #[allow(dead_code)]
    fn print_level_width_distribution(widths: &[usize]) {
        let n = widths.len();
        if n == 0 {
            println!("layer widths: (empty)");
            return;
        }
        let mut sorted = widths.to_vec();
        sorted.sort_unstable();
        let total: usize = widths.iter().sum();
        let min = *sorted.first().unwrap();
        let max = *sorted.last().unwrap();
        let mean = total as f64 / n as f64;
        let median = sorted[n / 2];
        let pct = |q: f64| sorted[((n as f64 * q) as usize).min(n - 1)];
        println!(
            "layer widths: n_levels={n} total_insts={total} min={min} max={max} mean={mean:.1} median={median}"
        );
        println!(
            "  p10={} p25={} p50={} p75={} p90={} p95={} p99={} p999={}",
            pct(0.10),
            pct(0.25),
            pct(0.50),
            pct(0.75),
            pct(0.90),
            pct(0.95),
            pct(0.99),
            pct(0.999),
        );
        // Log2-bucketed histogram: bucket `i` covers widths in [2^i, 2^(i+1)),
        // with a special zero bucket for width == 0.
        let max_bucket = ((max as f64).log2().floor() as usize + 1).min(32);
        let mut buckets = vec![0usize; max_bucket + 1];
        for &w in widths {
            let b = if w == 0 {
                0
            } else {
                (w as f64).log2().floor() as usize
            };
            let b = b.min(buckets.len() - 1);
            buckets[b] += 1;
        }
        let bar_width = 60usize;
        let max_count = buckets.iter().copied().max().unwrap_or(1).max(1);
        println!("  histogram (log2 buckets, [lo, hi) width range):");
        for (i, &count) in buckets.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let lo = if i == 0 { 0 } else { 1usize << i };
            let hi = 1usize << (i + 1);
            let bar_len = (count * bar_width) / max_count;
            let bar: String = "#".repeat(bar_len);
            let pct = count as f64 / n as f64 * 100.0;
            println!("    [{lo:>7}, {hi:>7}): {count:>6}  ({pct:>5.1}%) {bar}");
        }
    }

    #[test]
    #[ignore = "requires cached static verifier pk + root proof from bin/static-verifier-tracegen"]
    fn graph_executor_root_proof() {
        use std::time::Instant;

        let dir = std::env::var("STATIC_VERIFIER_CACHE_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../../bin/static-verifier-tracegen/cache")
            });
        let (circuit, shape): (StaticVerifierCircuit, crate::config::StaticVerifierShape) = {
            use std::io::Read as _;
            let mut reader = std::io::BufReader::new(
                std::fs::File::open(dir.join("static_verifier_pk.bin")).unwrap(),
            );
            let mut len_bytes = [0u8; 8];
            reader.read_exact(&mut len_bytes).unwrap();
            let mut bytes = vec![0u8; u64::from_le_bytes(len_bytes) as usize];
            reader.read_exact(&mut bytes).unwrap();
            serde_json::from_slice(&bytes).unwrap()
        };
        let proof: Proof<RootConfig> =
            bitcode::deserialize(&std::fs::read(dir.join("root_proof.bitcode")).unwrap()).unwrap();
        let log_heights_per_air = log_heights_per_air_from_proof(&proof);

        let start = Instant::now();
        let mut ir = Halo2IRBuilder::new(shape.lookup_bits);
        circuit.populate_verify_stark_constraints(&mut ir, &proof);
        println!("IR build: {:?}", start.elapsed());

        let start = Instant::now();
        let mut executor = GraphExecutor::new(ir);
        println!(
            "lowering: {:?} ({} insts, {} levels)",
            start.elapsed(),
            executor.insts.len(),
            executor.num_levels()
        );

        let start = Instant::now();
        load_proof_wire(&mut executor, &proof, &log_heights_per_air);
        println!("input population: {:?}", start.elapsed());

        let start = Instant::now();
        executor.run(1, |_, _| ());
        println!("run (1 thread): {:?}", start.elapsed());
        let reference_advice = executor.advice().to_vec();
        let reference_lookups = executor.lookups().to_vec();

        // Reruns reuse the warm tape (fresh-tape correctness vs the real backend
        // is covered by `graph_executor_matches_halo2_backend`); these are timing
        // plus write-offset consistency checks.
        for num_threads in [4, 8, 12] {
            let start = Instant::now();
            executor.run(num_threads, |_, _| ());
            println!("run ({num_threads} threads): {:?}", start.elapsed());
            assert_eq!(executor.advice(), &reference_advice[..]);
            assert_eq!(executor.lookups(), &reference_lookups[..]);
        }
    }

    /// Synthetic benchmark of the full witness-generation pipeline that
    /// [`StaticVerifierProvingKey::prove_wrapped`] runs before handing off to
    /// `snark_verifier_sdk`: `GraphProver::witness_gen` (parallel graph-IR
    /// evaluation) + the [`FusedColumnBuilder`] streaming per-column H2D
    /// copies. Stops after `builder.take_device_columns()` — SNARK generation
    /// itself is excluded.
    ///
    /// Loads the full [`StaticVerifierProvingKey`] and root proof from
    /// `bin/static-verifier-tracegen/cache/` (override with
    /// `STATIC_VERIFIER_CACHE_DIR=…`). Runs the pipeline three times per thread
    /// count so the first (cold) and subsequent (warm) numbers are visible.
    ///
    /// Run with:
    /// ```text
    /// cargo test --profile fast -p openvm-static-verifier \
    ///     --features evm-prove --release \
    ///     -- --ignored --nocapture \
    ///        graph_executor_prove_wrapped_pipeline
    /// ```
    #[test]
    #[cfg(feature = "evm-prove")]
    #[ignore = "requires cached static verifier pk + root proof + CUDA GPU"]
    fn graph_executor_prove_wrapped_pipeline() {
        use std::{fs::File, io::BufReader, path::PathBuf, time::Instant};

        use openvm_stark_sdk::openvm_stark_backend::codec::Decode as _;

        use crate::StaticVerifierProvingKey;

        let dir = std::env::var("STATIC_VERIFIER_CACHE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../../bin/static-verifier-tracegen/cache")
            });

        let start = Instant::now();
        let pk = {
            let mut reader =
                BufReader::new(File::open(dir.join("static_verifier_pk.bin")).unwrap());
            StaticVerifierProvingKey::decode(&mut reader).expect("decode static verifier pk")
        };
        println!("static_verifier_pk decode: {:?}", start.elapsed());

        let start = Instant::now();
        let proof: Proof<RootConfig> =
            bitcode::deserialize(&std::fs::read(dir.join("root_proof.bitcode")).unwrap()).unwrap();
        println!("root_proof decode: {:?}", start.elapsed());

        // Warm the GraphProver (IR build) so the timed loop measures only
        // per-proof work. The first pipeline call otherwise pays the lazy IR
        // build; subsequent calls reuse it.
        let start = Instant::now();
        let (warmup_advice, _) = pk.run_witness_gen_pipeline(&proof, 1, None);
        println!(
            "pipeline warm-up (1 thread, includes lazy IR build): {:?}",
            start.elapsed()
        );
        drop(warmup_advice);

        let level_widths = pk
            .graph_prover
            .get()
            .expect("graph prover initialized by warm-up call")
            .lock()
            .unwrap()
            .level_widths();
        print_level_width_distribution(&level_widths);

        for &num_threads in &[12usize] {
            for iter in 0..1 {
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
