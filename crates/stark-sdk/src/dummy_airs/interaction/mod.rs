use std::sync::Arc;

use ax_stark_backend::{
    keygen::MultiStarkKeygenBuilder,
    p3_matrix::dense::RowMajorMatrix,
    prover::{
        types::{AirProofInput, AirProofRawInput, ProofInput},
        MultiTraceStarkProver,
    },
    rap::AnyRap,
    verifier::{MultiTraceStarkVerifier, VerificationError},
};
use itertools::{izip, Itertools};
use p3_baby_bear::BabyBear;

use crate::config::{self, baby_bear_poseidon2::BabyBearPoseidon2Config};

pub mod dummy_interaction_air;

type Val = BabyBear;

pub fn verify_interactions(
    traces: Vec<RowMajorMatrix<Val>>,
    airs: Vec<Arc<dyn AnyRap<BabyBearPoseidon2Config>>>,
    pis: Vec<Vec<Val>>,
) -> Result<(), VerificationError> {
    let perm = config::baby_bear_poseidon2::random_perm();
    let config = config::baby_bear_poseidon2::default_config(&perm);

    let mut keygen_builder = MultiStarkKeygenBuilder::new(&config);
    let air_ids = airs
        .iter()
        .map(|air| keygen_builder.add_air(air.clone()))
        .collect_vec();
    let pk = keygen_builder.generate_pk();
    let vk = pk.get_vk();

    let per_air: Vec<_> = izip!(air_ids, airs, traces, pis)
        .map(|(air_id, air, trace, pvs)| {
            (
                air_id,
                AirProofInput {
                    air,
                    cached_mains_pdata: vec![],
                    raw: AirProofRawInput {
                        cached_mains: vec![],
                        common_main: Some(trace),
                        public_values: pvs,
                    },
                },
            )
        })
        .collect();

    let prover = MultiTraceStarkProver::new(&config);
    let mut challenger = config::baby_bear_poseidon2::Challenger::new(perm.clone());
    let proof = prover.prove(&mut challenger, &pk, ProofInput { per_air });

    // Verify the proof:
    // Start from clean challenger
    let mut challenger = config::baby_bear_poseidon2::Challenger::new(perm.clone());
    let verifier = MultiTraceStarkVerifier::new(prover.config);
    verifier.verify(&mut challenger, &vk, &proof)
}
