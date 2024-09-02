use ax_sdk::{
    config::{
        baby_bear_poseidon2::BabyBearPoseidon2Config, fri_params::default_fri_params, setup_tracing,
    },
    utils::{generate_fib_trace_rows, FibonacciAir},
};
use p3_field::AbstractField;
use p3_matrix::Matrix;
use p3_uni_stark::Val;

mod common;

#[test]
fn test_fibonacci() {
    type SC = BabyBearPoseidon2Config;
    type F = Val<SC>;

    setup_tracing();

    let fib_air = FibonacciAir {};
    let n = 16;
    let trace = generate_fib_trace_rows(n);
    let pvs = vec![vec![
        F::from_canonical_u32(0),
        F::from_canonical_u32(1),
        trace.get(n - 1, 1),
    ]];

    common::run_recursive_test(vec![&fib_air], vec![trace], pvs, default_fri_params())
}
