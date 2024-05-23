use itertools::Itertools;
use p3_field::AbstractField;
use p3_matrix::Matrix;
use rand::Rng;

use crate::config::{self, poseidon2::StarkConfigPoseidon2};
use afs_stark_backend::keygen::{types::SymbolicRap, MultiStarkKeygenBuilder};
use afs_stark_backend::prover::{
    trace::TraceCommitmentBuilder, types::ProverRap, MultiTraceStarkProver,
};
use afs_stark_backend::verifier::{types::VerifierRap, MultiTraceStarkVerifier, VerificationError};
use p3_baby_bear::BabyBear;
use p3_matrix::dense::DenseMatrix;
use p3_uni_stark::StarkGenericConfig;

pub trait ProverVerifierRap<SC: StarkGenericConfig>:
    ProverRap<SC> + VerifierRap<SC> + SymbolicRap<SC>
{
}
impl<SC: StarkGenericConfig, RAP: ProverRap<SC> + VerifierRap<SC> + SymbolicRap<SC>>
    ProverVerifierRap<SC> for RAP
{
}

// Returns row major matrix
pub fn generate_random_matrix<F: AbstractField>(
    mut rng: impl Rng,
    height: usize,
    width: usize,
) -> Vec<Vec<F>> {
    (0..height)
        .map(|_| {
            (0..width)
                .map(|_| F::from_wrapped_u32(rng.gen()))
                .collect_vec()
        })
        .collect_vec()
}

pub fn to_field_vec<F: AbstractField>(v: Vec<u32>) -> Vec<F> {
    v.into_iter().map(F::from_canonical_u32).collect()
}

/// This function assumes that all chips have no public inputs
pub fn run_simple_test(
    chips: Vec<&dyn ProverVerifierRap<StarkConfigPoseidon2>>,
    traces: Vec<DenseMatrix<BabyBear>>,
) -> Result<(), VerificationError> {
    assert_eq!(chips.len(), traces.len());

    let max_trace_length = traces.iter().map(|trace| trace.height()).max().unwrap_or(0);
    let log_trace_size = (max_trace_length as f64).log2().ceil() as usize;

    let perm = config::poseidon2::random_perm();
    let config = config::poseidon2::default_config(&perm, log_trace_size);

    let mut keygen_builder = MultiStarkKeygenBuilder::new(&config);

    for i in 0..chips.len() {
        keygen_builder.add_air(
            chips[i] as &dyn SymbolicRap<StarkConfigPoseidon2>,
            traces[i].height(),
            0,
        );
    }

    let pk = keygen_builder.generate_pk();
    let vk = pk.vk();

    let prover = MultiTraceStarkProver::new(config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.config.pcs());

    for trace in traces {
        trace_builder.load_trace(trace);
    }
    trace_builder.commit_current();

    let main_trace_data = trace_builder.view(
        &vk,
        chips
            .iter()
            .map(|&chip| chip as &dyn ProverRap<StarkConfigPoseidon2>)
            .collect(),
    );

    let pis = vec![vec![]; vk.per_air.len()];

    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let proof = prover.prove(&mut challenger, &pk, main_trace_data, &pis);

    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let verifier = MultiTraceStarkVerifier::new(prover.config);
    let result = verifier.verify(
        &mut challenger,
        vk,
        chips
            .iter()
            .map(|&chip| chip as &dyn VerifierRap<StarkConfigPoseidon2>)
            .collect(),
        proof,
        &pis,
    );

    result
}
