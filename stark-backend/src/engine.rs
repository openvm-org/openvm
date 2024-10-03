use itertools::izip;
use p3_matrix::dense::DenseMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};

use crate::{
    commit::SingleMatrixCommitPtr,
    config::{Com, PcsProof, PcsProverData},
    keygen::{types::MultiStarkVerifyingKey, MultiStarkKeygenBuilder},
    prover::{trace::TraceCommitmentBuilder, types::Proof, MultiTraceStarkProver},
    rap::AnyRap,
    verifier::{MultiTraceStarkVerifier, VerificationError},
};

/// Data for verifying a Stark proof.
pub struct VerificationData<SC: StarkGenericConfig> {
    pub vk: MultiStarkVerifyingKey<SC>,
    pub proof: Proof<SC>,
}

/// Testing engine
pub trait StarkEngine<SC: StarkGenericConfig> {
    /// Stark config
    fn config(&self) -> &SC;
    /// Creates a new challenger with a deterministic state.
    /// Creating new challenger for prover and verifier separately will result in
    /// them having the same starting state.
    fn new_challenger(&self) -> SC::Challenger;

    fn keygen_builder(&self) -> MultiStarkKeygenBuilder<SC> {
        MultiStarkKeygenBuilder::new(self.config())
    }

    fn trace_commitment_builder<'a>(&'a self) -> TraceCommitmentBuilder<'a, SC>
    where
        SC: 'a,
    {
        TraceCommitmentBuilder::new(self.config().pcs())
    }

    fn prover(&self) -> MultiTraceStarkProver<SC> {
        MultiTraceStarkProver::new(self.config())
    }

    fn verifier(&self) -> MultiTraceStarkVerifier<SC> {
        MultiTraceStarkVerifier::new(self.config())
    }

    /// Runs a single end-to-end test for a given set of chips and traces.
    /// This includes proving/verifying key generation, creating a proof, and verifying the proof.
    /// This function should only be used on chips where the main trace is **not** partitioned.
    ///
    /// - `chips`, `traces`, `public_values` should be zipped.
    fn run_simple_test(
        &self,
        chips: &[&dyn AnyRap<SC>],
        traces: Vec<DenseMatrix<Val<SC>>>,
        public_values: &[Vec<Val<SC>>],
    ) -> Result<VerificationData<SC>, VerificationError>
    where
        SC::Pcs: Sync,
        Domain<SC>: Send + Sync,
        PcsProverData<SC>: Send + Sync,
        Com<SC>: Send + Sync,
        SC::Challenge: Send + Sync,
        PcsProof<SC>: Send + Sync,
    {
        run_test_impl(
            self,
            chips,
            traces.into_iter().map(|t| vec![t]).collect(),
            public_values,
        )
    }

    /// Runs a single end-to-end test for a given set of chips and traces partitions.
    /// This includes proving/verifying key generation, creating a proof, and verifying the proof.
    ///
    /// - `chips`, `traces`, `public_values` should be zipped.
    fn run_test(
        &self,
        chips: &[&dyn AnyRap<SC>],
        traces: Vec<Vec<DenseMatrix<Val<SC>>>>,
        public_values: &[Vec<Val<SC>>],
    ) -> Result<VerificationData<SC>, VerificationError>
    where
        SC::Pcs: Sync,
        Domain<SC>: Send + Sync,
        PcsProverData<SC>: Send + Sync,
        Com<SC>: Send + Sync,
        SC::Challenge: Send + Sync,
        PcsProof<SC>: Send + Sync,
    {
        run_test_impl(self, chips, traces, public_values)
    }
}

fn run_test_impl<SC: StarkGenericConfig, E: StarkEngine<SC> + ?Sized>(
    engine: &E,
    chips: &[&dyn AnyRap<SC>],
    mut traces: Vec<Vec<DenseMatrix<Val<SC>>>>,
    public_values: &[Vec<Val<SC>>],
) -> Result<VerificationData<SC>, VerificationError>
where
    SC::Pcs: Sync,
    Domain<SC>: Send + Sync,
    PcsProverData<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Challenge: Send + Sync,
    PcsProof<SC>: Send + Sync,
{
    assert_eq!(chips.len(), traces.len());
    for (chip, chip_traces) in izip!(chips, traces.iter()) {
        // Note: we count the common main trace always even when its width is 0
        let num_traces = chip.cached_main_widths().len() + 1;
        assert_eq!(chip_traces.len(), num_traces);
    }

    let mut keygen_builder = engine.keygen_builder();

    let mut ptrs: Vec<Vec<SingleMatrixCommitPtr>> = vec![vec![]; traces.len()];
    // First, create pointers for cached traces
    for (chip, chip_ptrs) in izip!(chips, ptrs.iter_mut()) {
        let cached_trace_widths = chip.cached_main_widths();
        for width in cached_trace_widths {
            chip_ptrs.push(keygen_builder.add_cached_main_matrix(width));
        }
    }
    // Second, create pointer for common traces
    for (chip, chip_ptrs) in izip!(chips, ptrs.iter_mut()) {
        let common_trace_width = chip.common_main_width();
        chip_ptrs.push(keygen_builder.add_main_matrix(common_trace_width));
    }
    // Third, register all AIRs using the trace pointers
    for (chip, chip_ptrs) in izip!(chips, ptrs) {
        keygen_builder.add_partitioned_air(*chip, chip_ptrs);
    }

    let pk = keygen_builder.generate_pk();
    let vk = pk.vk();

    let prover = engine.prover();
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    // First, load all cached traces
    for (chip, chip_traces) in izip!(chips, traces.iter_mut()) {
        let num_cached_traces = chip.cached_main_widths().len();

        for _ in 0..num_cached_traces {
            let cached_trace = chip_traces.remove(0);
            let pdata = trace_builder.committer.commit(vec![cached_trace.clone()]);
            trace_builder.load_cached_trace(cached_trace, pdata);
        }
    }
    // Second, load the single common trace for each chip
    for chip_traces in traces.iter_mut() {
        trace_builder.load_trace(chip_traces.remove(0));
    }

    trace_builder.commit_current();

    let main_trace_data = trace_builder.view(
        &vk,
        chips.iter().map(|&chip| chip as &dyn AnyRap<SC>).collect(),
    );

    let mut challenger = engine.new_challenger();

    #[cfg(feature = "bench-metrics")]
    let prove_start = std::time::Instant::now();

    let proof = prover.prove(&mut challenger, &pk, main_trace_data, public_values);

    #[cfg(feature = "bench-metrics")]
    metrics::gauge!("stark_prove_excluding_trace_time_ms")
        .set(prove_start.elapsed().as_millis() as f64);

    let mut challenger = engine.new_challenger();
    let verifier = engine.verifier();
    verifier.verify(&mut challenger, &vk, &proof)?;
    Ok(VerificationData { vk, proof })
}
