use super::benchmark_helpers::run_recursive_test_benchmark;
use afs_test_utils::{
    config::{baby_bear_poseidon2::BabyBearPoseidon2Config, setup_tracing},
    utils::{generate_fib_trace_rows, FibonacciAir},
};
use p3_field::AbstractField;
use p3_matrix::Matrix;
use p3_uni_stark::Val;

pub fn benchmark_verify_fibair(n: usize) {
    println!("Running Verify Fibonacci Air benchmark with n = {}", n);

    type SC = BabyBearPoseidon2Config;
    type F = Val<SC>;

    setup_tracing();

    let fib_air = FibonacciAir {};
    let trace = generate_fib_trace_rows(n);
    let pvs = vec![vec![
        F::from_canonical_u32(0),
        F::from_canonical_u32(1),
        trace.get(n - 1, 1),
    ]];

    run_recursive_test_benchmark(vec![&fib_air], vec![&fib_air], vec![trace], pvs)
}
