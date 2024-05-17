use itertools::Itertools;
use p3_field::AbstractField;
use rand::Rng;

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
