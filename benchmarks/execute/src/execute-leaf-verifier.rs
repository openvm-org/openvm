use std::{fs, sync::Arc};

use clap::{arg, Parser, ValueEnum};
use eyre::Result;
use openvm_benchmarks_utils::get_fixtures_dir;
use openvm_circuit::arch::{instructions::exe::VmExe, ContinuationVmProof, VirtualMachine};
use openvm_continuations::verifier::root::types::RootVmVerifierInput;
use openvm_continuations::{
    verifier::{internal::types::InternalVmVerifierInput, leaf::types::LeafVmVerifierInput},
    SC,
};
use openvm_native_circuit::{NativeCpuBuilder, NATIVE_MAX_TRACE_HEIGHTS};
use openvm_sdk::config::{
    AggregationConfig, DEFAULT_NUM_CHILDREN_INTERNAL, DEFAULT_NUM_CHILDREN_LEAF,
};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Engine,
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

#[derive(Clone, Debug, ValueEnum)]
enum VerifierType {
    Leaf,
    Internal,
    Root,
}

#[derive(Parser)]
#[command(author, version, about = "OpenVM verifier execution")]
struct Cli {
    #[arg(short, long, value_enum, default_value = "preflight")]
    mode: ExecutionMode,

    #[arg(short, long, value_enum, default_value = "leaf")]
    verifier: VerifierType,

    #[arg(long, help = "Verifier index (for leaf and internal verifiers)")]
    index: Option<usize>,

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

    match cli.verifier {
        VerifierType::Leaf => {
            let leaf_exe_bytes =
                fs::read(fixtures_dir.join(format!("{}.leaf.exe", PROGRAM_NAME))).unwrap();
            let leaf_exe: VmExe<BabyBear> = bitcode::deserialize(&leaf_exe_bytes).unwrap();

            let leaf_pk_bytes =
                fs::read(fixtures_dir.join(format!("{}.leaf.pk", PROGRAM_NAME))).unwrap();
            let leaf_pk = bitcode::deserialize(&leaf_pk_bytes).unwrap();

            let leaf_inputs = LeafVmVerifierInput::chunk_continuation_vm_proof(
                &app_proof,
                DEFAULT_NUM_CHILDREN_LEAF,
            );
            let index = cli.index.unwrap_or(0);
            let leaf_input = leaf_inputs
                .get(index)
                .expect(&format!("No leaf input available at index {}", index));

            let agg_config = AggregationConfig::default();
            let config = agg_config.leaf_vm_config();
            let engine = BabyBearPoseidon2Engine::new(agg_config.leaf_fri_params);
            let d_pk = engine.device().transport_pk_to_device(&leaf_pk);
            let vm = VirtualMachine::new(engine, NativeCpuBuilder, config, d_pk)?;
            let input_stream = leaf_input.write_to_stream();

            execute_verifier(cli.mode, vm, leaf_exe, input_stream)?;
        }
        VerifierType::Internal => {
            let internal_exe_bytes =
                fs::read(fixtures_dir.join(format!("{}.internal.exe", PROGRAM_NAME))).unwrap();
            let internal_exe: VmExe<BabyBear> = bitcode::deserialize(&internal_exe_bytes).unwrap();

            let internal_pk_bytes =
                fs::read(fixtures_dir.join(format!("{}.internal.pk", PROGRAM_NAME))).unwrap();
            let internal_pk = bitcode::deserialize(&internal_pk_bytes).unwrap();

            // Create mock proofs for internal verifier (this would typically come from leaf proofs)
            let leaf_exe_bytes =
                fs::read(fixtures_dir.join(format!("{}.leaf.exe", PROGRAM_NAME))).unwrap();
            let leaf_exe: VmExe<BabyBear> = bitcode::deserialize(&leaf_exe_bytes).unwrap();
            let leaf_proofs = vec![]; // This would be generated from actual leaf proofs

            let internal_inputs = InternalVmVerifierInput::chunk_leaf_or_internal_proofs(
                Arc::new(internal_exe.program.clone()).into(),
                &leaf_proofs,
                DEFAULT_NUM_CHILDREN_INTERNAL,
            );
            let index = cli.index.unwrap_or(0);
            let internal_input = internal_inputs
                .get(index)
                .expect(&format!("No internal input available at index {}", index));

            let agg_config = AggregationConfig::default();
            let config = agg_config.internal_vm_config();
            let engine = BabyBearPoseidon2Engine::new(agg_config.internal_fri_params);
            let d_pk = engine.device().transport_pk_to_device(&internal_pk);
            let vm = VirtualMachine::new(engine, NativeCpuBuilder, config, d_pk)?;
            let input_stream = internal_input.write();

            execute_verifier(cli.mode, vm, internal_exe, input_stream)?;
        }
        VerifierType::Root => {
            let root_exe_bytes =
                fs::read(fixtures_dir.join(format!("{}.root.exe", PROGRAM_NAME))).unwrap();
            let root_exe: VmExe<BabyBear> = bitcode::deserialize(&root_exe_bytes).unwrap();

            let root_pk_bytes =
                fs::read(fixtures_dir.join(format!("{}.root.pk", PROGRAM_NAME))).unwrap();
            let root_pk = bitcode::deserialize(&root_pk_bytes).unwrap();

            // Create root verifier input
            let root_input = RootVmVerifierInput {
                proofs: vec![], // This would come from internal proofs
                public_values: app_proof.user_public_values.public_values.clone(),
            };

            let agg_config = AggregationConfig::default();
            let config = agg_config.root_vm_config();
            let engine = BabyBearPoseidon2Engine::new(agg_config.root_fri_params);
            let d_pk = engine.device().transport_pk_to_device(&root_pk);
            let vm = VirtualMachine::new(engine, NativeCpuBuilder, config, d_pk)?;
            let input_stream = root_input.write();

            execute_verifier(cli.mode, vm, root_exe, input_stream)?;
        }
    }

    Ok(())
}

fn execute_verifier(
    mode: ExecutionMode,
    vm: VirtualMachine<BabyBearPoseidon2Engine, NativeCpuBuilder>,
    exe: VmExe<BabyBear>,
    input_stream: impl Iterator<Item = BabyBear>,
) -> Result<()> {
    match mode {
        ExecutionMode::Normal => {
            tracing::info!("Running normal execute...");
            let interpreter = vm.executor().instance(&exe)?;
            interpreter.execute(input_stream, None)?;
        }
        ExecutionMode::Metered => {
            tracing::info!("Running metered execute...");
            let ctx = vm.build_metered_ctx();
            let interpreter = vm.metered_interpreter(&exe)?;
            interpreter.execute_metered(input_stream, ctx)?;
        }
        ExecutionMode::Preflight => {
            tracing::info!("Running preflight execute...");
            let state = vm.create_initial_state(&exe, input_stream);
            let mut interpreter = vm.preflight_interpreter(&exe)?;
            let _out = vm
                .execute_preflight(&mut interpreter, state, None, NATIVE_MAX_TRACE_HEIGHTS)
                .expect("Failed to execute preflight");
        }
    }
    Ok(())
}
