#![allow(unused_variables)]
#![allow(unused_imports)]
use std::rc::Rc;

use ax_stark_sdk::{
    bench::run_with_metric_collection,
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters},
    engine::StarkFriEngine,
    p3_baby_bear::BabyBear,
};
use axvm_benchmarks::utils::{bench_from_exe, build_bench_program, BenchmarkCli};
use axvm_circuit::arch::instructions::exe::AxVmExe;
use axvm_keccak256_circuit::Keccak256Rv32Config;
use axvm_keccak256_transpiler::Keccak256TranspilerExtension;
use axvm_native_circuit::NativeConfig;
use axvm_native_compiler::conversion::CompilerOptions;
use axvm_recursion::testing_utils::inner::build_verification_program;
use axvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use axvm_transpiler::{transpiler::Transpiler, FromElf};
use clap::Parser;
use eyre::Result;
use p3_field::AbstractField;
use tracing::info_span;

fn main() -> Result<()> {
    let cli_args = BenchmarkCli::parse();
    let app_log_blowup = cli_args.app_log_blowup.unwrap_or(2);
    let agg_log_blowup = cli_args.agg_log_blowup.unwrap_or(2);

    let elf = build_bench_program("base64_json")?;
    let exe = AxVmExe::from_elf(
        elf,
        Transpiler::<BabyBear>::default()
            .with_processor(Rc::new(Rv32ITranspilerExtension))
            .with_processor(Rc::new(Rv32MTranspilerExtension))
            .with_processor(Rc::new(Rv32IoTranspilerExtension))
            .with_processor(Rc::new(Keccak256TranspilerExtension)),
    );

    run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
        let vdata =
            info_span!("Base64 Json Program", group = "base64_json_program").in_scope(|| {
                let engine = BabyBearPoseidon2Engine::new(
                    FriParameters::standard_with_100_bits_conjectured_security(app_log_blowup),
                );

                let data = include_str!("../../programs/base64_json/json_payload_encoded.txt");

                let fe_bytes = data
                    .to_owned()
                    .into_bytes()
                    .into_iter()
                    .map(AbstractField::from_canonical_u8)
                    .collect::<Vec<BabyBear>>();
                bench_from_exe(engine, Keccak256Rv32Config::default(), exe, vec![fe_bytes])
            })?;

        #[cfg(feature = "aggregation")]
        {
            // Leaf aggregation: 1->1 proof "aggregation"
            let max_constraint_degree = ((1 << agg_log_blowup) + 1).min(7);
            let config = NativeConfig::aggregation(0, max_constraint_degree);
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
                        build_verification_program(vdata, compiler_options.clone());
                    let engine = BabyBearPoseidon2Engine::new(
                        FriParameters::standard_with_100_bits_conjectured_security(agg_log_blowup),
                    );
                    bench_from_exe(engine, config.clone(), program, input_stream).unwrap_or_else(
                        |e| panic!("Leaf aggregation failed for segment {}: {e}", seg_idx),
                    )
                });
            }
        }
        Ok(())
    })
}
