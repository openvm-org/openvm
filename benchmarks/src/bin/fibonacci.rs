#![allow(unused_variables)]
#![allow(unused_imports)]
use ax_stark_sdk::{
    bench::run_with_metric_collection,
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters},
    engine::StarkFriEngine,
};
use axvm_benchmarks::utils::{bench_from_exe, build_bench_program};
use axvm_circuit::arch::VmConfig;
use axvm_native_compiler::conversion::CompilerOptions;
use axvm_recursion::testing_utils::inner::build_verification_program;
use axvm_transpiler::axvm_platform::bincode;
use eyre::Result;
use p3_field::AbstractField;
use tracing::info_span;

fn main() -> Result<()> {
    // TODO[jpw]: benchmark different combinations
    let app_log_blowup = 1;
    let agg_log_blowup = 1;

    let elf = build_bench_program("fibonacci")?;
    run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
        let vdata =
            info_span!("Fibonacci Program", group = "fibonacci_program").in_scope(|| {
                let engine = BabyBearPoseidon2Engine::new(
                    FriParameters::standard_with_100_bits_conjectured_security(app_log_blowup),
                );
                let n = 100_000u64;
                let input = bincode::serde::encode_to_vec(n, bincode::config::standard())?;
                bench_from_exe(
                    engine,
                    VmConfig::rv32im(),
                    elf,
                    vec![input
                        .into_iter()
                        .map(AbstractField::from_canonical_u8)
                        .collect()],
                )
            })?;

        #[cfg(feature = "aggregation")]
        {
            // Leaf aggregation: 1->1 proof "aggregation"
            // TODO[jpw]: put real user public values number, placeholder=0
            let max_constraint_degree = ((1 << agg_log_blowup) + 1).min(7);
            let config = VmConfig::aggregation(0, max_constraint_degree);
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
