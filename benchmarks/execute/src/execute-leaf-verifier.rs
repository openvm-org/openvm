use std::fs;

use clap::{arg, Parser, ValueEnum};
use eyre::Result;
use openvm_benchmarks_utils::get_fixtures_dir;
use openvm_circuit::arch::{instructions::exe::VmExe, ContinuationVmProof, VirtualMachine};
use openvm_continuations::{
    verifier::{common::types::VmVerifierPvs, leaf::types::LeafVmVerifierInput},
    SC,
};
use openvm_native_circuit::{NativeConfig, NativeCpuBuilder, NATIVE_MAX_TRACE_HEIGHTS};
use openvm_sdk::config::{DEFAULT_LEAF_LOG_BLOWUP, SBOX_SIZE};
use openvm_stark_sdk::{
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters},
    engine::{StarkEngine, StarkFriEngine},
    openvm_stark_backend::prover::hal::DeviceDataTransporter,
    p3_baby_bear::BabyBear,
};
use tracing_subscriber::{fmt, EnvFilter};

const PROGRAM_NAME: &str = "kitchen-sink";

#[derive(Clone, Debug, ValueEnum)]
enum ExecutionMode {
    Normal,
    Metered,
    Preflight,
}

#[derive(Parser)]
#[command(author, version, about = "OpenVM leaf verifier execution")]
struct Cli {
    #[arg(short, long, value_enum, default_value = "preflight")]
    mode: ExecutionMode,

    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Set up logging
    let filter = if cli.verbose {
        EnvFilter::from_default_env()
    } else {
        EnvFilter::new("info")
    };
    fmt::fmt().with_env_filter(filter).init();

    let fixtures_dir = get_fixtures_dir();
    let app_proof_bytes =
        fs::read(fixtures_dir.join(format!("{}.app.proof", PROGRAM_NAME))).unwrap();
    let app_proof: ContinuationVmProof<SC> = bitcode::deserialize(&app_proof_bytes).unwrap();

    let leaf_exe_bytes = fs::read(fixtures_dir.join(format!("{}.leaf.exe", PROGRAM_NAME))).unwrap();
    let leaf_exe: VmExe<BabyBear> = bitcode::deserialize(&leaf_exe_bytes).unwrap();

    let leaf_pk_bytes = fs::read(fixtures_dir.join(format!("{}.leaf.pk", PROGRAM_NAME))).unwrap();
    let leaf_pk = bitcode::deserialize(&leaf_pk_bytes).unwrap();

    let leaf_inputs = LeafVmVerifierInput::chunk_continuation_vm_proof(&app_proof, 2);
    let leaf_input = leaf_inputs.first().expect("No leaf input available");

    let config = NativeConfig::aggregation(
        VmVerifierPvs::<u8>::width(),
        SBOX_SIZE.min(FriParameters::standard_fast().max_constraint_degree()),
    );
    let fri_params =
        FriParameters::standard_with_100_bits_conjectured_security(DEFAULT_LEAF_LOG_BLOWUP);
    let engine = BabyBearPoseidon2Engine::new(fri_params);
    let d_pk = engine.device().transport_pk_to_device(&leaf_pk);
    let vm = VirtualMachine::new(engine, NativeCpuBuilder, config, d_pk)?;
    let input_stream = leaf_input.write_to_stream();

    match cli.mode {
        ExecutionMode::Normal => {
            tracing::info!("Running normal execute...");
            let interpreter = vm.executor().instance(&leaf_exe)?;
            interpreter.execute(input_stream, None)?;
        }
        ExecutionMode::Metered => {
            tracing::info!("Running metered execute...");
            let ctx = vm.build_metered_ctx();
            let interpreter = vm.metered_interpreter(&leaf_exe)?;
            interpreter.execute_metered(input_stream, ctx)?;
        }
        ExecutionMode::Preflight => {
            tracing::info!("Running preflight execute...");
            let state = vm.create_initial_state(&leaf_exe, input_stream);
            let mut interpreter = vm.preflight_interpreter(&leaf_exe)?;
            let _out = vm
                .execute_preflight(&mut interpreter, state, None, NATIVE_MAX_TRACE_HEIGHTS)
                .expect("Failed to execute preflight");
        }
    }

    Ok(())
}
