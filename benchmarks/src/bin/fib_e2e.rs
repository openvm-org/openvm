use ax_stark_sdk::{
    bench::run_with_metric_collection,
    config::fri_params::standard_fri_params_with_100_bits_conjectured_security,
    p3_baby_bear::BabyBear,
};
use axvm_benchmarks::utils::{build_bench_program, BenchmarkCli};
use axvm_circuit::arch::instructions::{exe::AxVmExe, program::DEFAULT_MAX_NUM_PUBLIC_VALUES};
use axvm_native_compiler::conversion::CompilerOptions;
#[cfg(feature = "static-verifier")]
use axvm_native_compiler::prelude::Witness;
#[cfg(feature = "static-verifier")]
use axvm_recursion::witness::Witnessable;
use axvm_rv32im_circuit::Rv32ImConfig;
use axvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use axvm_sdk::{
    config::{AggConfig, AppConfig},
    e2e_prover::{commit_app_exe, generate_leaf_committed_exe, E2EStarkProver},
    keygen::{AggProvingKey, AppProvingKey},
};
use axvm_transpiler::{axvm_platform::bincode, transpiler::Transpiler, FromElf};
use clap::Parser;
use eyre::Result;
use p3_field::AbstractField;
#[cfg(feature = "static-verifier")]
use tracing::info_span;

type F = BabyBear;
const NUM_PUBLIC_VALUES: usize = DEFAULT_MAX_NUM_PUBLIC_VALUES;

#[tokio::main]
async fn main() -> Result<()> {
    let cli_args = BenchmarkCli::parse();
    let app_log_blowup = cli_args.app_log_blowup.unwrap_or(2);
    let agg_log_blowup = cli_args.agg_log_blowup.unwrap_or(2);
    let root_log_blowup = cli_args.root_log_blowup.unwrap_or(2);
    let internal_log_blowup = cli_args.internal_log_blowup.unwrap_or(2);

    // Must be larger than RangeTupleCheckerAir.height == 524288
    let segment_len = 1_000_000;

    let app_config = AppConfig {
        app_fri_params: standard_fri_params_with_100_bits_conjectured_security(app_log_blowup),
        app_vm_config: Rv32ImConfig::with_public_values_and_segment_len(
            NUM_PUBLIC_VALUES,
            segment_len,
        ),
    };
    let agg_config = AggConfig {
        max_num_user_public_values: NUM_PUBLIC_VALUES,
        leaf_fri_params: standard_fri_params_with_100_bits_conjectured_security(agg_log_blowup),
        internal_fri_params: standard_fri_params_with_100_bits_conjectured_security(
            internal_log_blowup,
        ),
        root_fri_params: standard_fri_params_with_100_bits_conjectured_security(root_log_blowup),
        compiler_options: CompilerOptions {
            enable_cycle_tracker: true,
            ..Default::default()
        },
    };

    let app_pk = AppProvingKey::keygen(app_config.clone());
    let agg_pk = AggProvingKey::keygen(agg_config.clone());
    let elf = build_bench_program("fibonacci")?;
    let exe = AxVmExe::from_elf(
        elf,
        Transpiler::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension),
    );
    let app_committed_exe = commit_app_exe(app_config, exe);
    let leaf_committed_exe = generate_leaf_committed_exe(agg_config, &app_pk);

    let prover = E2EStarkProver::new(app_pk, agg_pk, app_committed_exe, leaf_committed_exe, 2, 2);

    let n = 800_000u64;
    let app_input: Vec<_> = bincode::serde::encode_to_vec(n, bincode::config::standard())?
        .into_iter()
        .map(F::from_canonical_u8)
        .collect();
    run_with_metric_collection("OUTPUT_PATH", || {
        #[allow(unused_variables)]
        let root_proof =
            prover.generate_proof_with_metric_spans(app_input, "Fibonacci Continuation Program");
        #[cfg(feature = "static-verifier")]
        let static_verifier_snark = info_span!("static verifier", group = "static_verifier")
            .in_scope(|| {
                let static_verifier = prover
                    .agg_pk
                    .root_verifier_pk
                    .keygen_static_verifier(23, root_proof.clone());
                let mut witness = Witness::default();
                root_proof.write(&mut witness);
                static_verifier.prove(witness)
            });
    });

    Ok(())
}
