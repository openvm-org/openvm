#![allow(unused_variables)]
#![allow(unused_imports)]
use ax_stark_sdk::{
    bench::run_with_metric_collection,
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters},
    engine::StarkFriEngine,
};
use axvm_benchmarks::utils::{bench_from_exe, build_bench_program};
use axvm_circuit::arch::{ExecutorName, VmConfig};
use axvm_native_compiler::conversion::CompilerOptions;
use axvm_recursion::testing_utils::inner::build_verification_program;
use eyre::Result;
use p3_field::AbstractField;
use tracing::info_span;

fn main() -> Result<()> {
    // TODO[jpw]: benchmark different combinations
    let app_log_blowup = 1;
    // let agg_log_blowup = 1;

    // https://etherscan.io/tx/0xa40b41d5a00b8b1a1d591dc60882521942f5b98b277bb3e8ba6e0edda0a2e550
    let bytecode_hex = include_str!("../../programs/revm_contract_deployment/usdc/FiatTokenV1.bin");
    let bytecode = hex::decode(bytecode_hex)?;

    let elf = build_bench_program("revm_contract_deployment")?;
    run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
        let vdata = info_span!(
            "revm Contract Deployment",
            group = "revm_contract_deployment"
        )
        .in_scope(|| {
            let engine = BabyBearPoseidon2Engine::new(
                FriParameters::standard_with_100_bits_conjectured_security(app_log_blowup),
            );
            let bytecode = bytecode
                .into_iter()
                .map(AbstractField::from_canonical_u8)
                .collect();
            bench_from_exe(
                engine,
                VmConfig::rv32im().add_executor(ExecutorName::Keccak256Rv32),
                elf,
                vec![bytecode],
            )
        })?;
        Ok(())
    })
}
