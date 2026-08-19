//! Continuation proving expressed as a scheduler graph.
//!
//! One EXECUTE and one PROVE node per segment. EXECUTE nodes form a serial chain
//! because segment `n + 1`'s preflight starts from segment `n`'s output state.
//! PROVE nodes depend only on their own EXECUTE: a segment proof is a function of
//! the proving key and that segment's proving context alone, so PROVE nodes are
//! mutually independent and are where concurrency can come from.

pub use openvm_scheduler::{Budget, ResourceProfile};
use openvm_scheduler::{Engine, Node, SchedulerError};

/// GPU memory a prove pass needs no matter how many proves are resident.
///
/// Derived from the reference workload's device-wide peaks: 16 849 MiB with one
/// resident prove and 32 322 MiB with two, so the part that does not scale is
/// their difference subtracted from the single-prove peak.
pub const SHARED_GPU_BASE_BYTES: u64 = 1_376 << 20;

/// What one more resident PROVE adds on top of [`SHARED_GPU_BASE_BYTES`].
pub const PROVE_MARGINAL_GPU_BYTES: u64 = 15_473 << 20;

/// How far EXECUTE runs ahead of PROVE by default: far enough that two proves can
/// be resident at once, and no further.
pub const DEFAULT_PROVE_LOOKAHEAD: usize = 2;

/// Trace generation's own GPU residency has never been measured apart from the
/// prove it feeds, so it is left unclaimed rather than invented. Until it is
/// measured, admission is honest only about PROVE.
pub const EXECUTE_GPU_BYTES: u64 = 0;

/// A node of the continuation graph: a stage and the segment it belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SegmentNode {
    /// Preflight execution and trace generation for one segment.
    Execute(usize),
    /// The STARK prove for one segment.
    Prove(usize),
}

/// What each stage is declared to occupy, and the ceiling admission respects.
///
/// Only the GPU axis carries a number; host bytes and CPU threads are left at zero
/// rather than given invented ceilings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentSchedulerConfig {
    pub budget: Budget,
    pub execute: ResourceProfile,
    pub prove: ResourceProfile,
    /// How many segments EXECUTE may run ahead of PROVE.
    ///
    /// An EXECUTE hands its [`openvm_stark_backend::prover::ProvingContext`] to its
    /// PROVE and completes, so between those two points the traces are resident
    /// while no node holds budget for them. Admission cannot charge for that, so
    /// the number outstanding is bounded structurally instead: `E_n` also waits for
    /// `P_{n - lookahead}`. Below 2 the chain is fully serialized and no two proves
    /// are ever resident together.
    pub prove_lookahead: usize,
}

impl SegmentSchedulerConfig {
    /// Takes the budget as given: the caller decides what the machine allows.
    pub fn new(budget: Budget) -> Self {
        Self {
            budget,
            execute: ResourceProfile::new(EXECUTE_GPU_BYTES, 0, 0),
            prove: ResourceProfile::new(PROVE_MARGINAL_GPU_BYTES, 0, 0),
            prove_lookahead: DEFAULT_PROVE_LOOKAHEAD,
        }
    }

    /// Budget for a device with `device_gpu_bytes` of memory in total, measured
    /// the way `nvidia-smi` reports it — driver and CUDA context included.
    ///
    /// Admission is additive: it compares the sum of resident profiles against the
    /// budget, so it cannot express a cost that is paid once however many nodes are
    /// resident. The shared base is therefore withheld from the budget instead of
    /// being declared on a node, leaving per-node profiles carrying only what each
    /// additional resident prove adds. That is exact while at least one prove is
    /// resident and conservative by the base when none is.
    pub fn for_device(device_gpu_bytes: u64) -> Self {
        let admissible = device_gpu_bytes.saturating_sub(SHARED_GPU_BASE_BYTES);
        Self::new(Budget::new(admissible, 0, 0))
    }
}

/// Registers the EXECUTE/PROVE nodes and edges for a metered segmentation.
///
/// Dependencies must already be registered when a node is added, so each segment
/// contributes its EXECUTE before its PROVE and the graph is acyclic by
/// construction.
pub(crate) fn segment_graph(
    num_segments: usize,
    config: &SegmentSchedulerConfig,
) -> Result<Engine<SegmentNode>, SchedulerError<SegmentNode>> {
    let mut engine = Engine::new(config.budget);
    let lookahead = config.prove_lookahead.max(1);
    for idx in 0..num_segments {
        let mut predecessors = Vec::new();
        if idx > 0 {
            predecessors.push(SegmentNode::Execute(idx - 1));
        }
        if let Some(bounded_by) = idx.checked_sub(lookahead) {
            predecessors.push(SegmentNode::Prove(bounded_by));
        }
        engine.add_node(Node::new(
            SegmentNode::Execute(idx),
            predecessors,
            config.execute,
        ))?;
        engine.add_node(Node::new(
            SegmentNode::Prove(idx),
            vec![SegmentNode::Execute(idx)],
            config.prove,
        ))?;
    }
    Ok(engine)
}
