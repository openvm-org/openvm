use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{AbstractField, Field, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_uni_stark::Val;

use afs_stark_backend::interaction::AirBridge;
use afs_test_utils::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use afs_test_utils::config::setup_tracing;

mod common;

pub struct FibonacciAir;

impl<F: Field> AirBridge<F> for FibonacciAir {}

impl<F> BaseAir<F> for FibonacciAir {
    fn width(&self) -> usize {
        2
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

pub fn generate_trace_rows<F: PrimeField32>(n: usize) -> RowMajorMatrix<F> {
    assert!(n.is_power_of_two());

    let mut rows = vec![vec![F::zero(), F::one()]];

    for i in 1..n {
        rows.push(vec![rows[i - 1][1], rows[i - 1][0] + rows[i - 1][1]]);
    }

    RowMajorMatrix::new(rows.concat(), 2)
}

#[test]
fn test_fibonacci() {
    type SC = BabyBearPoseidon2Config;
    type F = Val<SC>;

    setup_tracing();

    let fib_air = FibonacciAir {};
    let n = 16;
    let trace = generate_trace_rows(n);
    let pvs = vec![vec![
        F::from_canonical_u32(0),
        F::from_canonical_u32(1),
        trace.get(n - 1, 1),
    ]];

    common::run_recursive_test(vec![&fib_air], vec![&fib_air], vec![trace], pvs)
}
