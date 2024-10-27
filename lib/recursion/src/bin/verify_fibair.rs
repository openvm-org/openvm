/// Benchmark of aggregation VM performance.
/// Proofs:
/// 1. Prove Fibonacci AIR.
/// 2. Verify the proof of 1. by execution VM program in STARK VM.
use afs_compiler::conversion::CompilerOptions;
use afs_recursion::testing_utils::recursive_stark_test;
use ax_stark_backend::Chip;
use ax_sdk::{
    bench::run_with_metric_collection,
    config::{
        baby_bear_poseidon2::BabyBearPoseidon2Engine,
        fri_params::standard_fri_params_with_100_bits_conjectured_security,
    },
    dummy_airs::fib_air::chip::FibonacciChip,
    engine::StarkFriEngine,
};
use stark_vm::system::vm::config::VmConfig;

fn main() {
    run_with_metric_collection("OUTPUT_PATH", || {
        let n = 16; // STARK to calculate 16th Fibonacci number.
        let fib_chip = FibonacciChip::new(0, 1, n);
        let vdata =
            BabyBearPoseidon2Engine::run_test_fast(vec![fib_chip.generate_air_proof_input()])
                .unwrap();

        let compiler_options = CompilerOptions {
            enable_cycle_tracker: true,
            ..Default::default()
        };
        recursive_stark_test(
            vdata,
            compiler_options,
            VmConfig::aggregation(4, 7),
            &BabyBearPoseidon2Engine::new(standard_fri_params_with_100_bits_conjectured_security(
                3,
            )),
        )
        .unwrap();
    });
}
