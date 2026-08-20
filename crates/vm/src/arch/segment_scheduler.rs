//! Continuation proving expressed as a scheduler graph.
//!
//! One EXECUTE and one PROVE node per segment. EXECUTE nodes form a serial chain
//! because segment `n + 1`'s preflight starts from segment `n`'s output state.
//! PROVE nodes depend only on their own EXECUTE: a segment proof is a function of
//! the proving key and that segment's proving context alone, so PROVE nodes are
//! mutually independent and are where concurrency can come from.

use openvm_scheduler::{Admission, Engine, Node, SchedulerError};
pub use openvm_scheduler::{Budget, ResourceProfile};

use crate::arch::{execution_mode::Segment, VirtualMachineError};

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

    /// How many proves this budget admits at once.
    ///
    /// A backend that gives each admitted prove its own device queue sizes its
    /// pool from this, so the pool and admission agree on the ceiling instead of
    /// each carrying its own copy of it. Always at least one: a budget too small
    /// for a single prove still has to run it.
    pub fn max_resident_proves(&self) -> usize {
        if self.prove.gpu_bytes == 0 {
            return 1;
        }
        usize::try_from(self.budget.gpu_bytes / self.prove.gpu_bytes)
            .unwrap_or(usize::MAX)
            .max(1)
    }
}

/// A graph with no segments in it yet.
pub(crate) fn empty_graph(config: &SegmentSchedulerConfig) -> Engine<SegmentNode> {
    Engine::new(config.budget)
}

/// Adds one segment's EXECUTE and PROVE nodes and their edges.
///
/// Segments are registered in order, so every dependency named here is already
/// registered and the graph is acyclic by construction. Callers that discover
/// segments as they go can keep extending the graph while earlier nodes run.
pub(crate) fn register_segment(
    engine: &mut Engine<SegmentNode>,
    idx: usize,
    config: &SegmentSchedulerConfig,
) -> Result<(), SchedulerError<SegmentNode>> {
    let mut predecessors = Vec::new();
    if idx > 0 {
        predecessors.push(SegmentNode::Execute(idx - 1));
    }
    if let Some(bounded_by) = idx.checked_sub(config.prove_lookahead.max(1)) {
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
    ))
}

/// What a proving backend must supply for the graph to be driven against it.
///
/// The graph decides *when* each half of a segment runs; this decides *how*. Both
/// continuation drivers implement it, so the admission policy exists once.
///
/// Implemented outside this crate — the record-free CUDA driver in
/// `openvm-sdk-config` is one — so **adding a method is a breaking change** for any
/// implementor pinned at a different revision. Extend it with a defaulted method,
/// or with a separate trait, rather than a bare addition. The same applies to
/// [`SegmentSource`].
pub trait SegmentDriver {
    /// A segment's proving input, produced by an execute and consumed by its prove.
    type Ctx;
    type Proof;

    /// Advances the execute chain by one segment and yields its proving context.
    ///
    /// Executes are a serial chain — segment `n + 1` starts from segment `n`'s
    /// output state — so this takes the driver mutably and runs where it is
    /// admitted, on the caller's thread.
    fn execute(&mut self, idx: usize, segment: &Segment) -> Result<Self::Ctx, VirtualMachineError>;

    /// Proves every entry of `batch` concurrently, running `while_proving` on the
    /// calling thread meanwhile, and returns the proofs alongside whatever
    /// `while_proving` produced.
    ///
    /// Each entry must be proved on its own device queue: seating two proves
    /// together buys nothing if their work serializes behind one queue. Arranging
    /// that is the backend's concern, which is why concurrency lives here and not
    /// in the caller.
    fn prove_batch(
        &self,
        batch: Vec<(usize, Self::Ctx)>,
        while_proving: &mut dyn FnMut() -> Result<Vec<Segment>, VirtualMachineError>,
    ) -> Result<ProvedBatch<Self::Proof>, VirtualMachineError>;
}

/// What one call to [`SegmentDriver::prove_batch`] returns: the batch's proofs
/// paired with their segment indices, and whatever the producer yielded meanwhile.
pub type ProvedBatch<P> = (Vec<(usize, P)>, Vec<Segment>);

/// Yields the metered segmentation, ideally as it is discovered rather than only
/// once the whole program has been segmented.
pub trait SegmentSource {
    fn is_finished(&self) -> bool;

    /// Advances to the next boundary and returns whatever segments that closed.
    fn step(&mut self) -> Result<Vec<Segment>, VirtualMachineError>;
}

/// One scheduled continuation run, with the observations a correctness test needs
/// to tell a scheduled run from a serial one.
pub struct ScheduledRun<P> {
    /// Proofs in segment order, which is not the order they completed in.
    pub proofs: Vec<P>,
    pub segments: Vec<Segment>,
    /// High-water mark of proves dispatched together.
    pub max_concurrent_proves: usize,
}

/// Runs one continuation proof by admitting graph nodes against `config`'s budget.
///
/// A segment proof is a function of the proving key and that segment's context
/// alone, and proofs are assembled by segment index rather than completion order,
/// so what this returns does not depend on the order the graph admits work in.
pub fn drive_scheduled<D, S>(
    config: &SegmentSchedulerConfig,
    driver: &mut D,
    source: &mut S,
) -> Result<ScheduledRun<D::Proof>, VirtualMachineError>
where
    D: SegmentDriver,
    S: SegmentSource,
{
    let mut graph = empty_graph(config);
    let mut segments: Vec<Segment> = Vec::new();
    let mut contexts: Vec<Option<D::Ctx>> = Vec::new();
    let mut proofs: Vec<Option<D::Proof>> = Vec::new();
    // Admitted proves wait here rather than running where they were admitted, so
    // that everything the budget lets in at once goes to the backend together. An
    // execute is not held: it runs where it is admitted, because it needs the
    // driver mutably.
    let mut waiting: Vec<(usize, D::Ctx)> = Vec::new();
    let mut max_concurrent_proves = 0usize;
    let mut proved = 0usize;
    loop {
        // Only enough to get started, or to break an idle graph. The bulk of
        // production happens while proves are in flight, below.
        if segments.is_empty() && !source.is_finished() {
            let fresh = source.step()?;
            register_segments(&mut graph, &mut segments, fresh, config)?;
            contexts.resize_with(segments.len(), || None);
            proofs.resize_with(segments.len(), || None);
        }
        match graph.admit() {
            Admission::Admitted(nodes) => {
                for node in nodes {
                    match node {
                        SegmentNode::Execute(idx) => {
                            contexts[idx] = Some(driver.execute(idx, &segments[idx])?);
                            // Execute holds nothing past its own run; releasing it
                            // here is what lets its prove and the next execute in.
                            graph.complete(&node).map_err(scheduling_error)?;
                        }
                        SegmentNode::Prove(idx) => {
                            let ctx = contexts[idx]
                                .take()
                                .expect("a prove node is admitted only after its execute node");
                            waiting.push((idx, ctx));
                        }
                    }
                }
            }
            // Only a prove is ever left holding budget, so both of these mean at
            // least one is waiting and running them is what makes progress.
            Admission::Backpressure | Admission::Blocked => {
                let batch = std::mem::take(&mut waiting);
                assert!(
                    !batch.is_empty(),
                    "admission reports backpressure or blocked only while a prove is resident"
                );
                max_concurrent_proves = max_concurrent_proves.max(batch.len());
                // How far production may run ahead while these proves work.
                // Bounded, because each discovered segment's execute will hold a
                // proving context until its prove consumes it.
                let target = proved + config.prove_lookahead + 2;
                let produced_len = segments.len();
                let (results, fresh) = driver.prove_batch(batch, &mut || {
                    let mut fresh = Vec::new();
                    while !source.is_finished() && produced_len + fresh.len() < target {
                        fresh.extend(source.step()?);
                    }
                    Ok(fresh)
                })?;
                register_segments(&mut graph, &mut segments, fresh, config)?;
                contexts.resize_with(segments.len(), || None);
                proofs.resize_with(segments.len(), || None);
                for (idx, proof) in results {
                    proofs[idx] = Some(proof);
                    proved += 1;
                    graph
                        .complete(&SegmentNode::Prove(idx))
                        .map_err(scheduling_error)?;
                }
            }
            // Every registered node is done. That is only the end once the source
            // has no more segments to register.
            Admission::AllComplete => {
                if source.is_finished() && segments.len() == proved {
                    break;
                }
                // Idle with work still to discover: there is no prove in flight to
                // overlap against, so produce here.
                let fresh = source.step()?;
                register_segments(&mut graph, &mut segments, fresh, config)?;
                contexts.resize_with(segments.len(), || None);
                proofs.resize_with(segments.len(), || None);
            }
        }
    }
    Ok(ScheduledRun {
        proofs: proofs
            .into_iter()
            .map(|proof| proof.expect("every registered prove node ran"))
            .collect(),
        segments,
        max_concurrent_proves,
    })
}

/// Adds every segment of `fresh` to the graph and to `segments`.
fn register_segments(
    graph: &mut Engine<SegmentNode>,
    segments: &mut Vec<Segment>,
    fresh: Vec<Segment>,
    config: &SegmentSchedulerConfig,
) -> Result<(), VirtualMachineError> {
    for segment in fresh {
        register_segment(graph, segments.len(), config).map_err(scheduling_error)?;
        segments.push(segment);
    }
    Ok(())
}

fn scheduling_error(error: impl std::fmt::Display) -> VirtualMachineError {
    VirtualMachineError::Scheduling(error.to_string())
}
