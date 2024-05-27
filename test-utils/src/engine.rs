use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData},
    keygen::{types::SymbolicRap, MultiStarkKeygenBuilder},
    prover::{trace::TraceCommitmentBuilder, types::ProverRap, MultiTraceStarkProver},
    verifier::{types::VerifierRap, MultiTraceStarkVerifier, VerificationError},
};
use p3_matrix::{dense::DenseMatrix, Matrix};
use p3_uni_stark::{Domain, StarkGenericConfig, Val};

use crate::{config::instrument::StarkHashStatistics, utils::ProverVerifierRap};

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

    fn run_simple_test(
        &self,
        chips: Vec<&dyn ProverVerifierRap<SC>>,
        traces: Vec<DenseMatrix<Val<SC>>>,
    ) -> Result<(), VerificationError>
    where
        SC::Pcs: Sync,
        Domain<SC>: Send + Sync,
        PcsProverData<SC>: Send + Sync,
        Com<SC>: Send + Sync,
        SC::Challenge: Send + Sync,
        PcsProof<SC>: Send + Sync,
    {
        run_simple_test_impl(self, chips, traces)
    }
}

pub trait StarkEngineWithHashInstrumentation<SC: StarkGenericConfig>: StarkEngine<SC> {
    fn clear_instruments(&mut self);
    fn stark_hash_statistics<T>(&self, custom: T) -> StarkHashStatistics<T>;
}

/// This function assumes that all chips have no public inputs
fn run_simple_test_impl<SC: StarkGenericConfig, E: StarkEngine<SC> + ?Sized>(
    engine: &E,
    chips: Vec<&dyn ProverVerifierRap<SC>>,
    traces: Vec<DenseMatrix<Val<SC>>>,
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
        keygen_builder.add_air(chips[i] as &dyn SymbolicRap<SC>, traces[i].height(), 0);
    }

    let pk = keygen_builder.generate_pk();
    let vk = pk.vk();

    let prover = engine.prover();
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    for trace in traces {
        trace_builder.load_trace(trace);
    }
    trace_builder.commit_current();

    let main_trace_data = trace_builder.view(
        &vk,
        chips
            .iter()
            .map(|&chip| chip as &dyn ProverRap<SC>)
            .collect(),
    );

    let pis = vec![vec![]; vk.per_air.len()];

    let mut challenger = engine.new_challenger();
    let proof = prover.prove(&mut challenger, &pk, main_trace_data, &pis);

    let mut challenger = engine.new_challenger();
    let verifier = engine.verifier();
    verifier.verify(
        &mut challenger,
        vk,
        chips
            .iter()
            .map(|&chip| chip as &dyn VerifierRap<SC>)
            .collect(),
        proof,
        &pis,
    )
}
