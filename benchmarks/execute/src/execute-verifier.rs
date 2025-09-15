//! OpenVM verifier execution
//!
//! First, generate the fixtures by running:
//! ```bash
//! cargo r -r --bin generate-fixtures --features generate-fixtures
//! ```
//!
//! To profile this binary, build it with the profiling profile:
//! ```bash
//! cargo b --profile profiling --bin execute-verifier
//! ```
//!
//! Then run it with samply for profiling:
//! ```bash
//! samply record --rate 10000 target/profiling/execute-verifier --mode preflight --verifier internal kitchen-sink
//! ```

use std::fs;

use clap::{arg, Parser, ValueEnum};
use eyre::Result;
use openvm_benchmarks_utils::get_fixtures_dir;
use openvm_circuit::arch::{
    instructions::exe::VmExe, ContinuationVmProof, Streams, VirtualMachine,
};
#[cfg(feature = "evm-prove")]
use openvm_continuations::verifier::root::types::RootVmVerifierInput;
use openvm_continuations::{
    verifier::{internal::types::InternalVmVerifierInput, leaf::types::LeafVmVerifierInput},
    SC,
};
use openvm_native_circuit::{NativeCpuBuilder, NATIVE_MAX_TRACE_HEIGHTS};
use openvm_native_recursion::hints::Hintable;
use openvm_sdk::{
    commit::VmCommittedExe,
    config::{AggregationConfig, DEFAULT_NUM_CHILDREN_INTERNAL, DEFAULT_NUM_CHILDREN_LEAF},
};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Engine,
    engine::{StarkEngine, StarkFriEngine},
    openvm_stark_backend::{
        config::StarkGenericConfig, proof::Proof, prover::hal::DeviceDataTransporter,
    },
    p3_baby_bear::BabyBear,
};
use tracing_subscriber::{fmt, EnvFilter};

#[derive(Clone, Debug, ValueEnum)]
enum ExecutionMode {
    Pure,
    Metered,
    Preflight,
}

#[derive(Clone, Debug, ValueEnum)]
enum VerifierType {
    Leaf,
    Internal,
    #[cfg(feature = "evm-prove")]
    Root,
}

#[derive(Clone, Debug, ValueEnum)]
enum ProofType {
    Leaf,
    Internal,
}

#[derive(Parser)]
#[command(author, version, about = "OpenVM verifier execution")]
struct Cli {
    /// Program name to use for fixtures
    #[arg(value_name = "PROGRAM", default_value = "kitchen-sink")]
    program: String,

    #[arg(short, long, value_enum, default_value = "leaf")]
    verifier: VerifierType,

    #[arg(short, long, value_enum, default_value = "preflight")]
    mode: ExecutionMode,

    #[arg(long, help = "Verifier index (for leaf and internal verifiers)")]
    index: Option<usize>,

    #[arg(short, long)]
    verbose: bool,
}

fn load_proof(
    fixtures_dir: &std::path::Path,
    program_name: &str,
    proof_type: ProofType,
    index: usize,
) -> Result<Proof<SC>> {
    let proof_filename = match proof_type {
        ProofType::Leaf => format!("{}.leaf.{}.proof", program_name, index),
        ProofType::Internal => format!("{}.internal.{}.proof", program_name, index),
    };

    let proof_bytes = fs::read(fixtures_dir.join(proof_filename))
        .unwrap_or_else(|_| panic!("No {:?} proof available at index {}", proof_type, index));
    let proof: Proof<SC> = bitcode::deserialize(&proof_bytes).unwrap();
    Ok(proof)
}

/// Determines which proofs an internal verifier at given index should aggregate
/// Returns (proof_type, indices) where indices are the proof indices to load
fn get_internal_verifier_proof_indices(
    internal_verifier_index: usize,
    leaf_proof_count: usize,
    num_children: usize,
) -> (ProofType, Vec<usize>) {
    // Calculate how internal proofs are generated in layers
    let mut current_layer_proofs = leaf_proof_count;
    let mut current_layer_start_idx = 0;
    let mut is_leaf_layer = true;
    let mut internal_node_idx = 0;

    // First, traverse through regular aggregation layers
    loop {
        // Calculate number of internal proofs in current layer
        let num_internal_proofs_in_layer = current_layer_proofs.div_ceil(num_children);

        if internal_node_idx + num_internal_proofs_in_layer > internal_verifier_index {
            // Found the layer containing our internal verifier
            let layer_local_idx = internal_verifier_index - internal_node_idx;
            let start_proof_idx = layer_local_idx * num_children;
            let end_proof_idx = ((layer_local_idx + 1) * num_children).min(current_layer_proofs);

            let proof_indices: Vec<usize> = (start_proof_idx..end_proof_idx)
                .map(|i| {
                    if is_leaf_layer {
                        i
                    } else {
                        current_layer_start_idx + i
                    }
                })
                .collect();

            let proof_type = if is_leaf_layer {
                ProofType::Leaf
            } else {
                ProofType::Internal
            };

            return (proof_type, proof_indices);
        }

        // Move to next layer
        internal_node_idx += num_internal_proofs_in_layer;
        if !is_leaf_layer {
            current_layer_start_idx += current_layer_proofs;
        }
        current_layer_proofs = num_internal_proofs_in_layer;
        is_leaf_layer = false;

        // If we're down to 1 proof, we've reached the final internal proof
        if current_layer_proofs == 1 {
            break;
        }
    }

    // If we get here, the index might be for a wrapper internal proof
    // Wrapper proofs aggregate exactly one proof (the previous internal proof)
    let last_regular_internal_idx = internal_node_idx;
    if internal_verifier_index >= last_regular_internal_idx {
        // This is a wrapper proof - it aggregates exactly one internal proof
        // The wrapper at index `last_regular_internal_idx` wraps the final internal proof (at index
        // last_regular_internal_idx - 1) The wrapper at index `last_regular_internal_idx +
        // 1` wraps the wrapper at index last_regular_internal_idx
        let wrapper_layer = internal_verifier_index - last_regular_internal_idx;
        let proof_to_wrap_idx = if wrapper_layer == 0 {
            last_regular_internal_idx - 1 // Wrap the final internal proof
        } else {
            internal_verifier_index - 1 // Wrap the previous wrapper
        };

        return (ProofType::Internal, vec![proof_to_wrap_idx]);
    }

    panic!(
        "Invalid internal verifier index {}",
        internal_verifier_index
    );
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
        fs::read(fixtures_dir.join(format!("{}.app.proof", cli.program))).unwrap();
    let app_proof: ContinuationVmProof<SC> = bitcode::deserialize(&app_proof_bytes).unwrap();

    match cli.verifier {
        VerifierType::Leaf => {
            let leaf_exe_bytes =
                fs::read(fixtures_dir.join(format!("{}.leaf.exe", cli.program))).unwrap();
            let leaf_exe: VmExe<BabyBear> = bitcode::deserialize(&leaf_exe_bytes).unwrap();

            let leaf_pk_bytes =
                fs::read(fixtures_dir.join(format!("{}.leaf.pk", cli.program))).unwrap();
            let leaf_pk = bitcode::deserialize(&leaf_pk_bytes).unwrap();

            let leaf_inputs = LeafVmVerifierInput::chunk_continuation_vm_proof(
                &app_proof,
                DEFAULT_NUM_CHILDREN_LEAF,
            );
            let index = cli.index.unwrap_or(0);
            let leaf_input = leaf_inputs
                .get(index)
                .unwrap_or_else(|| panic!("No leaf input available at index {}", index));

            let agg_config = AggregationConfig::default();
            let config = agg_config.leaf_vm_config();
            let engine = BabyBearPoseidon2Engine::new(agg_config.leaf_fri_params);
            let d_pk = engine.device().transport_pk_to_device(&leaf_pk);
            let vm = VirtualMachine::new(engine, NativeCpuBuilder, config, d_pk)?;
            let input_stream = leaf_input.write_to_stream();

            execute_verifier(cli.mode, vm, &leaf_exe, input_stream)?;
        }
        VerifierType::Internal => {
            let internal_exe_bytes =
                fs::read(fixtures_dir.join(format!("{}.internal.exe", cli.program))).unwrap();
            let internal_exe: VmExe<BabyBear> = bitcode::deserialize(&internal_exe_bytes).unwrap();

            let internal_pk_bytes =
                fs::read(fixtures_dir.join(format!("{}.internal.pk", cli.program))).unwrap();
            let internal_pk = bitcode::deserialize(&internal_pk_bytes).unwrap();

            let index = cli.index.unwrap_or(0);

            // Count available leaf proofs
            let leaf_proof_count = {
                let mut count = 0;
                while fixtures_dir
                    .join(format!("{}.leaf.{}.proof", cli.program, count))
                    .exists()
                {
                    count += 1;
                }
                count
            };

            // Determine which proofs this internal verifier should aggregate
            let (proof_type, proof_indices) = get_internal_verifier_proof_indices(
                index,
                leaf_proof_count,
                DEFAULT_NUM_CHILDREN_INTERNAL,
            );

            // Load the determined proofs
            let proofs: Vec<_> = proof_indices
                .into_iter()
                .map(|idx| load_proof(&fixtures_dir, &cli.program, proof_type.clone(), idx))
                .collect::<Result<Vec<_>, _>>()?;

            tracing::info!(
                "Internal verifier {} will aggregate {} {:?} proofs",
                index,
                proofs.len(),
                proof_type
            );

            let agg_config = AggregationConfig::default();
            let config = agg_config.internal_vm_config();
            let engine = BabyBearPoseidon2Engine::new(agg_config.internal_fri_params);

            let internal_committed_exe =
                VmCommittedExe::<SC>::commit(internal_exe, engine.config().pcs());
            let internal_inputs = InternalVmVerifierInput::chunk_leaf_or_internal_proofs(
                internal_committed_exe.get_program_commit().into(),
                &proofs,
                DEFAULT_NUM_CHILDREN_INTERNAL,
            );
            let d_pk = engine.device().transport_pk_to_device(&internal_pk);
            let vm = VirtualMachine::new(engine, NativeCpuBuilder, config, d_pk)?;
            let input_stream = internal_inputs.first().unwrap().write();

            execute_verifier(cli.mode, vm, &internal_committed_exe.exe, input_stream)?;
        }
        #[cfg(feature = "evm-prove")]
        VerifierType::Root => {
            let root_exe_bytes =
                fs::read(fixtures_dir.join(format!("{}.root.exe", cli.program))).unwrap();
            let root_exe: VmExe<BabyBear> = bitcode::deserialize(&root_exe_bytes).unwrap();

            let root_pk_bytes =
                fs::read(fixtures_dir.join(format!("{}.root.pk", cli.program))).unwrap();
            let root_pk = bitcode::deserialize(&root_pk_bytes).unwrap();

            // Load root verifier input
            let root_input_bytes =
                fs::read(fixtures_dir.join(format!("{}.root.input", cli.program))).unwrap();
            let root_input: RootVmVerifierInput<SC> =
                bitcode::deserialize(&root_input_bytes).unwrap();

            let agg_config = AggregationConfig::default();
            let config = agg_config.root_verifier_vm_config();
            let engine = BabyBearPoseidon2Engine::new(agg_config.root_fri_params);
            let d_pk = engine.device().transport_pk_to_device(&root_pk);
            let vm = VirtualMachine::new(engine, NativeCpuBuilder, config, d_pk)?;
            let input_stream = root_input.write();

            execute_verifier(cli.mode, vm, &root_exe, input_stream)?;
        }
    }

    Ok(())
}

fn execute_verifier(
    mode: ExecutionMode,
    vm: VirtualMachine<BabyBearPoseidon2Engine, NativeCpuBuilder>,
    exe: &VmExe<BabyBear>,
    input_stream: impl Into<Streams<BabyBear>>,
) -> Result<()> {
    match mode {
        ExecutionMode::Pure => {
            tracing::info!("Running pure execute...");
            let interpreter = vm.executor().instance(exe)?;
            interpreter.execute(input_stream, None)?;
        }
        ExecutionMode::Metered => {
            tracing::info!("Running metered execute...");
            let ctx = vm.build_metered_ctx(exe);
            let interpreter = vm.metered_interpreter(exe)?;
            interpreter.execute_metered(input_stream, ctx)?;
        }
        ExecutionMode::Preflight => {
            tracing::info!("Running preflight execute...");
            let state = vm.create_initial_state(exe, input_stream);
            let mut interpreter = vm.preflight_interpreter(exe)?;
            vm.execute_preflight(&mut interpreter, state, None, NATIVE_MAX_TRACE_HEIGHTS)?;
        }
    }
    Ok(())
}
