use afs_stark_backend::rap::{BaseAirWithPublicValues, PartitionedBaseAir};
use itertools::Itertools;
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{AbstractField, PrimeField};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rand::{rngs::StdRng, Rng, SeedableRng};

pub struct FibonacciAir;

impl<F> PartitionedBaseAir<F> for FibonacciAir {}
impl<F> BaseAir<F> for FibonacciAir {
    fn width(&self) -> usize {
        2
    }
}

impl<F> BaseAirWithPublicValues<F> for FibonacciAir {
    fn num_public_values(&self) -> usize {
        3
    }
}

impl<AB: AirBuilderWithPublicValues> Air<AB> for FibonacciAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let pis = builder.public_values();

        let a = pis[0];
        let b = pis[1];
        let x = pis[2];

        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let mut when_first_row = builder.when_first_row();
        when_first_row.assert_eq(local[0], a);
        when_first_row.assert_eq(local[1], b);

        let mut when_transition = builder.when_transition();
        when_transition.assert_eq(next[0], local[1]);
        when_transition.assert_eq(next[1], local[0] + local[1]);

        builder.when_last_row().assert_eq(local[1], x);
    }
}

pub fn generate_fib_trace_rows<F: PrimeField>(n: usize) -> RowMajorMatrix<F> {
    assert!(n.is_power_of_two());

    let mut rows = vec![vec![F::zero(), F::one()]];

    for i in 1..n {
        rows.push(vec![rows[i - 1][1], rows[i - 1][0] + rows[i - 1][1]]);
    }

    RowMajorMatrix::new(rows.concat(), 2)
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

/// A macro to create a `Vec<&dyn AnyRap<_>>` from a list of AIRs because Rust cannot infer the
/// type correctly when using `vec!`.
#[macro_export]
macro_rules! any_rap_vec {
    [$($e:expr),*] => {
        {
            let chips: Vec<&dyn afs_stark_backend::rap::AnyRap<_>> = vec![$($e),*];
            chips
        }
    };
}

/// A macro to create a `Vec<Box<dyn AnyRap<_>>>` from a list of AIRs because Rust cannot infer the
/// type correctly when using `vec!`.
#[macro_export]
macro_rules! any_rap_box_vec {
    [$($e:expr),*] => {
        {
            let chips: Vec<Box<dyn afs_stark_backend::rap::AnyRap<_>>> = vec![$(Box::new($e)),*];
            chips
        }
    };
}
