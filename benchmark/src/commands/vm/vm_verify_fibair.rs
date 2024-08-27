use afs_test_utils::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Config,
    utils::{generate_fib_trace_rows, FibonacciAir},
};
use color_eyre::eyre::Result;
use p3_field::AbstractField;
use p3_matrix::Matrix;
use p3_uni_stark::Val;

use super::benchmark_helpers::run_recursive_test_benchmark;
use crate::utils::tracing::setup_benchmark_tracing;

pub fn benchmark_verify_fibair(n: usize) -> Result<()> {
    println!("Running Verify Fibonacci Air benchmark with n = {}", n);

    type SC = BabyBearPoseidon2Config;
    type F = Val<SC>;

    setup_benchmark_tracing();

    let fib_air = FibonacciAir {};
    let trace = generate_fib_trace_rows(n);
    let pvs = vec![vec![
        F::from_canonical_u32(0),
        F::from_canonical_u32(1),
        trace.get(n - 1, 1),
    ]];

    run_recursive_test_benchmark(
        vec![&fib_air],
        vec![trace],
        pvs,
        "VM Verifier for Fibonacci Air",
    )
}
