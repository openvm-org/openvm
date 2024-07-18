use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData},
    keygen::MultiStarkKeygenBuilder,
    prover::{trace::TraceCommitmentBuilder, MultiTraceStarkProver},
    rap::AnyRap,
    verifier::{MultiTraceStarkVerifier, VerificationError},
};
use p3_matrix::dense::DenseMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};

use crate::config::instrument::StarkHashStatistics;

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
        chips: Vec<&dyn AnyRap<SC>>,
        traces: Vec<DenseMatrix<Val<SC>>>,
        public_values: Vec<Vec<Val<SC>>>,
    ) -> Result<(), VerificationError>
    where
        SC::Pcs: Sync,
        Domain<SC>: Send + Sync,
        PcsProverData<SC>: Send + Sync,
        Com<SC>: Send + Sync,
        SC::Challenge: Send + Sync,
        PcsProof<SC>: Send + Sync,
    {
        run_simple_test_impl(self, chips, traces, public_values)
    }
}

pub trait StarkEngineWithHashInstrumentation<SC: StarkGenericConfig>: StarkEngine<SC> {
    fn clear_instruments(&mut self);
    fn stark_hash_statistics<T>(&self, custom: T) -> StarkHashStatistics<T>;
}

fn run_simple_test_impl<SC: StarkGenericConfig, E: StarkEngine<SC> + ?Sized>(
    engine: &E,
    chips: Vec<&dyn AnyRap<SC>>,
    traces: Vec<DenseMatrix<Val<SC>>>,
    public_values: Vec<Vec<Val<SC>>>,
) -> Result<(), VerificationError>
where
    SC::Pcs: Sync,
    Domain<SC>: Send + Sync,
    PcsProverData<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Challenge: Send + Sync,
    PcsProof<SC>: Send + Sync,
{
    assert_eq!(chips.len(), traces.len());

    let mut keygen_builder = engine.keygen_builder();

    for i in 0..chips.len() {
        keygen_builder.add_air(chips[i] as &dyn AnyRap<SC>, public_values[i].len());
    }

    let partial_pk = keygen_builder.generate_partial_pk();
    let partial_vk = partial_pk.partial_vk();

    let prover = engine.prover();
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    for trace in traces {
        trace_builder.load_trace(trace);
    }
    trace_builder.commit_current();

    let main_trace_data = trace_builder.view(
        &partial_vk,
        chips.iter().map(|&chip| chip as &dyn AnyRap<SC>).collect(),
    );

    let mut challenger = engine.new_challenger();
    let proof = prover.prove(
        &mut challenger,
        &partial_pk,
        main_trace_data,
        &public_values,
    );

    let mut challenger = engine.new_challenger();
    let verifier = engine.verifier();
    verifier.verify(&mut challenger, &partial_vk, chips, &proof, &public_values)
}
