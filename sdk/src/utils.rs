use itertools::Itertools;
use p3_field::AbstractField;
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Deterministic seeded RNG, for testing use
pub fn create_seeded_rng() -> StdRng {
    let seed = [42; 32];
    StdRng::from_seed(seed)
}

pub fn create_seeded_rng_with_seed(seed: u64) -> StdRng {
    let seed_be = seed.to_be_bytes();
    let mut seed = [0u8; 32];
    seed[24..32].copy_from_slice(&seed_be);
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

/// A macro to create a `Vec<Arc<dyn AnyRap<_>>>` from a list of AIRs because Rust cannot infer the
/// type correctly when using `vec!`.
#[macro_export]
macro_rules! any_rap_arc_vec {
    [$($e:expr),*] => {
        {
            let chips: Vec<std::sync::Arc<dyn afs_stark_backend::rap::AnyRap<_>>> = vec![$(std::sync::Arc::new($e)),*];
            chips
        }
    };
}
