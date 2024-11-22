#![allow(unused_variables)]
#![allow(unused_imports)]

use ax_stark_sdk::{
    bench::run_with_metric_collection,
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters},
    engine::StarkFriEngine,
};
use axvm_benchmarks::utils::{bench_from_exe, build_bench_program, BenchmarkCli};
use axvm_circuit::arch::{instructions::exe::AxVmExe, ExecutorName, VmConfig};
use axvm_native_compiler::conversion::CompilerOptions;
use axvm_recursion::testing_utils::inner::build_verification_program;
use axvm_transpiler::axvm_platform::bincode;
use clap::Parser;
use eyre::Result;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use tracing::info_span;

fn main() -> Result<()> {
    let cli_args = BenchmarkCli::parse();
    let app_log_blowup = cli_args.app_log_blowup.unwrap_or(2);
    let agg_log_blowup = cli_args.agg_log_blowup.unwrap_or(2);

    let elf = build_bench_program("ecrecover")?;
    let exe = AxVmExe::<BabyBear>::from(elf.clone());
    let vm_config = VmConfig::rv32im()
        .add_executor(ExecutorName::Keccak256Rv32)
        .add_modular_support(exe.custom_op_config.primes())
        .add_canonical_ec_curves(); // TODO: update sw_setup macros and read it from elf.

    run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
        let vdata =
            info_span!("ECDSA Recover Program", group = "ecrecover_program").in_scope(|| {
                let engine = BabyBearPoseidon2Engine::new(
                    FriParameters::standard_with_100_bits_conjectured_security(app_log_blowup),
                );
                let msg = (0..1u8).collect::<Vec<_>>();
                let input = bincode::serde::encode_to_vec(msg, bincode::config::standard())?;
                bench_from_exe(
                    engine,
                    vm_config,
                    elf,
                    vec![input
                        .into_iter()
                        .map(AbstractField::from_canonical_u8)
                        .collect()],
                )
            })?;

        Ok(())
    })
}
