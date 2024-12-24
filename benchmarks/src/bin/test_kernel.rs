#![allow(unused_variables)]
#![allow(unused_imports)]

use clap::Parser;
use eyre::Result;
use openvm_benchmarks::utils::{bench_from_exe, build_bench_program, BenchmarkCli};
use openvm_circuit::arch::{instructions::exe::VmExe, VmConfig};
use openvm_edsl_deserialize::{
    config::Rv32WithKernelsConfig, deserialize_instruction::LongFormTranspilerExtension,
};
use openvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use openvm_sdk::{config::AppConfig, StdIn};
use openvm_stark_sdk::{
    config::{
        baby_bear_poseidon2::BabyBearPoseidon2Engine,
        fri_params::standard_fri_params_with_100_bits_conjectured_security, FriParameters,
    },
    engine::StarkFriEngine,
    p3_baby_bear::BabyBear,
};
use openvm_transpiler::{transpiler::Transpiler, FromElf};

fn main() -> Result<()> {
    let cli_args = BenchmarkCli::parse();
    let app_log_blowup = cli_args.app_log_blowup.unwrap_or(2);
    let agg_log_blowup = cli_args.agg_log_blowup.unwrap_or(2);

    let elf = build_bench_program("test_kernel")?;
    let exe = VmExe::from_elf(
        elf,
        Transpiler::<BabyBear>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension)
            .with_extension(LongFormTranspilerExtension),
    )?;

    let app_fri_params = standard_fri_params_with_100_bits_conjectured_security(
        cli_args.app_log_blowup.unwrap_or(2),
    );
    let app_config = AppConfig::new(app_fri_params, Rv32WithKernelsConfig::default());

    bench_from_exe("test_kernel", app_config, exe, StdIn::default(), false)?;

    Ok(())

    /*run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
        let vdata =
            info_span!("Base64 Json Program", group = "base64_json_program").in_scope(|| {
                let engine = BabyBearPoseidon2Engine::new(
                    FriParameters::standard_with_100_bits_conjectured_security(app_log_blowup),
                );

                let data = include_str!("../../programs/base64_json/json_payload_encoded.txt");

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
    })*/
}
