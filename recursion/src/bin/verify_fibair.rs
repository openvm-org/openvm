/// Benchmark of aggregation VM performance.
/// Proofs:
/// 1. Prove Fibonacci AIR.
/// 2. Verify the proof of 1. by execution VM program in STARK VM.
use afs_compiler::conversion::CompilerOptions;
use afs_recursion::testing_utils::recursive_stark_test;
use ax_sdk::{
    bench::run_with_metric_collection,
    config::{
        baby_bear_poseidon2::BabyBearPoseidon2Engine,
        fri_params::fri_params_with_80_bits_of_security,
    },
    engine::StarkFriEngine,
    utils::{generate_fib_trace_rows, FibonacciAir},
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::Matrix;
use stark_vm::vm::config::VmConfig;

fn main() {
    run_with_metric_collection("OUTPUT_PATH", || {
        let n = 16; // STARK to calculate 16th Fibonacci number.
        let fib_air = FibonacciAir {};
        let trace = generate_fib_trace_rows(n); // n rows
        let pvs = vec![vec![
            BabyBear::from_canonical_u32(0),
            BabyBear::from_canonical_u32(1),
            trace.get(n - 1, 1),
        ]];
        let vdata =
            BabyBearPoseidon2Engine::run_simple_test(&[&fib_air], vec![trace], &pvs).unwrap();

        let compiler_options = CompilerOptions {
            enable_cycle_tracker: true,
            ..Default::default()
        };
        recursive_stark_test(
            vdata,
            pvs,
            compiler_options,
            VmConfig::aggregation(7),
            BabyBearPoseidon2Engine::new(fri_params_with_80_bits_of_security()[1]),
        )
        .unwrap();
    });
}
