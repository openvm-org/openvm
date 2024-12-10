#![allow(unused_variables)]
#![allow(unused_imports)]

use ax_stark_backend::p3_field::AbstractField;
use ax_stark_sdk::{
    bench::run_with_metric_collection,
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters},
    engine::StarkFriEngine,
    p3_baby_bear::BabyBear,
};
use axvm_benchmarks::utils::{bench_from_exe, build_bench_program, BenchmarkCli};
use axvm_circuit::arch::instructions::{exe::AxVmExe, program::DEFAULT_MAX_NUM_PUBLIC_VALUES};
use axvm_keccak256_circuit::Keccak256Rv32Config;
use axvm_keccak256_transpiler::Keccak256TranspilerExtension;
use axvm_native_circuit::NativeConfig;
use axvm_native_compiler::conversion::CompilerOptions;
use axvm_native_recursion::testing_utils::inner::build_verification_program;
use axvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use axvm_sdk::StdIn;
use axvm_transpiler::{transpiler::Transpiler, FromElf};
use clap::Parser;
use eyre::Result;
use tracing::info_span;

fn main() -> Result<()> {
    let cli_args = BenchmarkCli::parse();
    let app_log_blowup = cli_args.app_log_blowup.unwrap_or(2);
    let agg_log_blowup = cli_args.agg_log_blowup.unwrap_or(2);

    let elf = build_bench_program("regex")?;
    let exe = AxVmExe::from_elf(
        elf.clone(),
        Transpiler::<BabyBear>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension)
            .with_extension(Keccak256TranspilerExtension),
    )?;
    run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
        let vdata = info_span!("Regex Program").in_scope(|| {
            let engine = BabyBearPoseidon2Engine::new(
                FriParameters::standard_with_100_bits_conjectured_security(app_log_blowup),
            );

            let data = include_str!("../../programs/regex/regex_email.txt");

            let fe_bytes = data.to_owned().into_bytes();
            bench_from_exe(
                engine,
                Keccak256Rv32Config::default(),
                exe,
                StdIn::from_bytes(&fe_bytes),
            )
        })?;

        #[cfg(feature = "aggregation")]
        {
            // Leaf aggregation: 1->1 proof "aggregation"
            // TODO[jpw]: put real user public values number, placeholder=0
            let max_constraint_degree = ((1 << agg_log_blowup) + 1).min(7);
            let config =
                NativeConfig::aggregation(DEFAULT_MAX_NUM_PUBLIC_VALUES, max_constraint_degree)
                    .with_continuations();
            let compiler_options = CompilerOptions {
                enable_cycle_tracker: true,
                ..Default::default()
            };
            for (seg_idx, vdata) in vdata.into_iter().enumerate() {
                info_span!(
                    "Leaf Aggregation",
                    group = "leaf_aggregation",
                    segment = seg_idx
                )
                .in_scope(|| {
                    let (program, input_stream) =
                        build_verification_program(vdata, compiler_options);
                    let engine = BabyBearPoseidon2Engine::new(
                        FriParameters::standard_with_100_bits_conjectured_security(agg_log_blowup),
                    );
                    bench_from_exe(engine, config.clone(), program, input_stream.into())
                        .unwrap_or_else(|e| {
                            panic!("Leaf aggregation failed for segment {}: {e}", seg_idx)
                        })
                });
            }
        }
        Ok(())
    })
}