use std::{fs, path::Path};

use eyre::Result;
use openvm_benchmarks_utils::{get_elf_path, get_fixtures_dir, get_programs_dir, read_elf_file};
use openvm_circuit::arch::{
    PreflightExecutor, SingleSegmentVmProver, VmBuilder, VmExecutionConfig,
};
use openvm_continuations::verifier::internal::types::InternalVmVerifierInput;
use openvm_native_circuit::{NativeConfig, NATIVE_MAX_TRACE_HEIGHTS};
use openvm_native_recursion::hints::Hintable;
use openvm_sdk::{
    config::{AggregationTreeConfig, AppConfig, AppFriParams, SdkVmConfig},
    prover::AggStarkProver,
    CpuSdk, StdIn, F, SC,
};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Engine, engine::StarkFriEngine,
    openvm_stark_backend::proof::Proof,
};
use serde::Serialize;
use tracing::info_span;
use tracing_subscriber::{fmt, EnvFilter};

const PROGRAMS_AND_INPUTS: &[(&str, Option<u64>)] =
    &[("fibonacci", Some(1 << 22)), ("kitchen-sink", None)];

/// Helper function to serialize and write data to a file
fn write_fixture<T: Serialize>(path: impl AsRef<Path>, data: &T, description: &str) -> Result<()> {
    tracing::debug!("Writing {}", description);
    let bytes = bitcode::serialize(data)?;
    fs::write(path, bytes)?;
    Ok(())
}

/// Aggregates leaf proofs into internal proofs and saves them to disk
/// Returns the final internal proof and the count of internal proofs saved
fn aggregate_leaf_proofs<E, NativeBuilder>(
    agg_prover: &mut AggStarkProver<E, NativeBuilder>,
    leaf_proofs: Vec<Proof<SC>>,
    fixtures_dir: &Path,
    program: &str,
) -> Result<(Proof<SC>, usize)>
where
    E: StarkFriEngine<SC = SC>,
    NativeBuilder: VmBuilder<E, VmConfig = NativeConfig>,
    <NativeConfig as VmExecutionConfig<F>>::Executor:
        PreflightExecutor<F, <NativeBuilder as VmBuilder<E>>::RecordArena>,
{
    let mut internal_node_idx = -1;
    let mut internal_node_height = 0;
    let mut internal_proof_count = 0;
    let mut proofs = leaf_proofs;

    // We will always generate at least one internal proof, even if there is only one leaf
    // proof, in order to shrink the proof size
    while proofs.len() > 1 || internal_node_height == 0 {
        let internal_inputs = InternalVmVerifierInput::chunk_leaf_or_internal_proofs(
            (*agg_prover.internal_prover.program_commitment()).into(),
            &proofs,
            agg_prover.num_children_internal,
        );

        proofs = info_span!(
            "agg_layer",
            group = format!("internal.{internal_node_height}")
        )
        .in_scope(|| -> Result<Vec<Proof<SC>>> {
            internal_inputs
                .into_iter()
                .enumerate()
                .map(|(i, input)| -> Result<Proof<SC>> {
                    internal_node_idx += 1;
                    let proof = info_span!("single_internal_agg", idx = internal_node_idx)
                        .in_scope(|| {
                            SingleSegmentVmProver::prove(
                                &mut agg_prover.internal_prover,
                                input.write(),
                                NATIVE_MAX_TRACE_HEIGHTS,
                            )
                        })?;

                    // Save proof
                    write_fixture(
                        fixtures_dir.join(format!(
                            "{}.internal.{}.proof",
                            program,
                            internal_proof_count + i
                        )),
                        &proof,
                        &format!("internal proof {}", internal_proof_count + i),
                    )?;

                    Ok(proof)
                })
                .collect::<Result<Vec<_>, _>>()
        })?;

        internal_proof_count += proofs.len();
        internal_node_height += 1;
    }

    let final_internal_proof = proofs.pop().unwrap();
    Ok((final_internal_proof, internal_proof_count))
}

fn main() -> Result<()> {
    // Set up logging
    fmt::fmt().with_env_filter(EnvFilter::new("info")).init();

    // Create fixtures directory if it doesn't exist
    let fixtures_dir = get_fixtures_dir();
    fs::create_dir_all(&fixtures_dir)?;

    tracing::info!("Processing {} programs", PROGRAMS_AND_INPUTS.len());

    for (idx, &(program, input)) in PROGRAMS_AND_INPUTS.iter().enumerate() {
        tracing::info!(
            "Processing program {}/{}: {} {}",
            idx + 1,
            PROGRAMS_AND_INPUTS.len(),
            program,
            input
                .map(|i| format!("(input: {})", i))
                .unwrap_or_else(|| "(no input)".to_string())
        );

        let program_dir = get_programs_dir().join(program);

        tracing::info!(program = %program, "Loading VM config");
        let config_path = program_dir.join("openvm.toml");
        let config_content = fs::read_to_string(&config_path)?;
        let vm_config = SdkVmConfig::from_toml(&config_content)?.app_vm_config;

        tracing::info!(program = %program, "Preparing ELF");
        let elf_path = get_elf_path(&program_dir);
        let elf = read_elf_file(&elf_path)?;

        // Create app config with default parameters
        let app_config = AppConfig::new(AppFriParams::default().fri_params, vm_config);

        let sdk = CpuSdk::new(app_config.clone())?;
        let exe = sdk.convert_to_exe(elf)?;

        // Prepare stdin
        let mut stdin = StdIn::default();
        if let Some(input_value) = input {
            tracing::info!(program = %program, input = %input_value, "Preparing stdin with input");
            stdin.write(&input_value);
        } else {
            tracing::info!(program = %program, "No input provided for program");
        }

        tracing::info!(program = %program, "Generating app proof");
        let app_proof = sdk.app_prover(exe)?.prove(stdin)?;

        // Save app proof
        write_fixture(
            fixtures_dir.join(format!("{}.app.proof", program)),
            &app_proof,
            "app proof",
        )?;

        tracing::info!(program = %program, "Getting keys");
        let app_pk = sdk.app_pk();
        let agg_pk = sdk.agg_pk();

        // Save keys
        write_fixture(
            fixtures_dir.join(format!("{}.leaf.exe", program)),
            &app_pk.leaf_committed_exe.exe,
            "leaf exe",
        )?;

        write_fixture(
            fixtures_dir.join(format!("{}.leaf.pk", program)),
            &agg_pk.leaf_vm_pk.vm_pk,
            "leaf proving key",
        )?;

        write_fixture(
            fixtures_dir.join(format!("{}.internal.exe", program)),
            &agg_pk.internal_committed_exe.exe,
            "internal exe",
        )?;

        write_fixture(
            fixtures_dir.join(format!("{}.internal.pk", program)),
            &agg_pk.internal_vm_pk.vm_pk,
            "internal proving key",
        )?;

        tracing::info!(program = %program, "Creating aggregation provers");
        let native_builder = sdk.native_builder().clone();
        let leaf_verifier_exe = app_pk.leaf_committed_exe.exe.clone();

        let tree_config = AggregationTreeConfig::default();
        let mut agg_prover = AggStarkProver::<BabyBearPoseidon2Engine, _>::new(
            native_builder.clone(),
            agg_pk,
            leaf_verifier_exe,
            tree_config,
        )?;

        tracing::info!(program = %program, "Generating leaf proofs");
        let leaf_proofs = agg_prover.generate_leaf_proofs(&app_proof)?;
        tracing::info!(program = %program, leaf_proof_count = leaf_proofs.len(), "Generated leaf proofs");

        // Save leaf proofs
        for (i, leaf_proof) in leaf_proofs.iter().enumerate() {
            write_fixture(
                fixtures_dir.join(format!("{}.leaf.{}.proof", program, i)),
                leaf_proof,
                &format!("leaf proof {}", i),
            )?;
        }

        tracing::info!(program = %program, "Generating internal proofs");

        let (_final_internal_proof, internal_proof_count) =
            aggregate_leaf_proofs(&mut agg_prover, leaf_proofs.clone(), &fixtures_dir, program)?;

        tracing::info!(
            program = %program,
            leaf_proofs = leaf_proofs.len(),
            total_internals = internal_proof_count,
            "Generated and saved {} fixtures: leaf.exe, leaf.pk, internal.exe, internal.pk, app.proof, {} leaf proofs, and {} internal proofs",
            program,
            leaf_proofs.len(),
            internal_proof_count
        );
    }

    tracing::info!("Successfully processed all programs");
    Ok(())
}
