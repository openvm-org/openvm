use itertools::Itertools;
use p3_field::AbstractField;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use afs_stark_backend::keygen::types::SymbolicRap;
use afs_stark_backend::prover::types::ProverRap;
use afs_stark_backend::verifier::types::VerifierRap;
use p3_uni_stark::StarkGenericConfig;

pub trait ProverVerifierRap<SC: StarkGenericConfig>:
    ProverRap<SC> + VerifierRap<SC> + SymbolicRap<SC>
{
}
impl<SC: StarkGenericConfig, RAP: ProverRap<SC> + VerifierRap<SC> + SymbolicRap<SC>>
    ProverVerifierRap<SC> for RAP
{
}

/// Deterministic seeded RNG, for testing use
pub fn create_seeded_rng() -> StdRng {
    let seed = [42; 32];
    StdRng::from_seed(seed)
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
