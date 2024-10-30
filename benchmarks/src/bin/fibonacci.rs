use ax_stark_sdk::{
    bench::run_with_metric_collection,
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters},
    engine::StarkFriEngine,
};
use axvm_benchmarks::utils::{bench_from_elf, build_bench_program};
use axvm_circuit::arch::VmConfig;
use eyre::Result;
use tracing::info_span;

fn main() -> Result<()> {
    let engine = BabyBearPoseidon2Engine::new(FriParameters::standard_fast());
    let elf = build_bench_program("fibonacci")?;
    let config = VmConfig::rv32im();
    run_with_metric_collection("OUTPUT_PATH", || {
        let _vdata = info_span!("Fibonacci Program", group = "fibonacci_program")
            .in_scope(|| bench_from_elf(engine, config, elf, vec![]));
    });

    Ok(())
}
