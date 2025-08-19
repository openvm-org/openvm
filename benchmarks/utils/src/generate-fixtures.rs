use std::fs;

use eyre::Result;
use openvm_benchmarks_utils::{get_elf_path, get_fixtures_dir, get_programs_dir, read_elf_file};
use openvm_circuit::arch::{
    PreflightExecutor, SingleSegmentVmProver, VmBuilder, VmExecutionConfig,
};
use openvm_continuations::verifier::internal::types::{InternalVmVerifierInput, VmStarkProof};
use openvm_native_circuit::{NativeConfig, NATIVE_MAX_TRACE_HEIGHTS};
use openvm_native_recursion::hints::Hintable;
use openvm_sdk::{
    codec::Encode,
    config::{AggregationTreeConfig, AppConfig, AppFriParams, SdkVmConfig},
    prover::AggStarkProver,
    Sdk, StdIn, F, SC,
};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Engine, engine::StarkFriEngine,
    openvm_stark_backend::proof::Proof,
};
use tracing::info_span;
use tracing_subscriber::{fmt, EnvFilter};

const PROGRAM: &str = "kitchen-sink";

fn aggregate_leaf_proofs_and_collect_internals<E, NativeBuilder>(
    agg_prover: &mut AggStarkProver<E, NativeBuilder>,
    leaf_proofs: Vec<Proof<SC>>,
    public_values: Vec<F>,
) -> Result<(VmStarkProof<SC>, Vec<Proof<SC>>)>
where
    E: StarkFriEngine<SC = SC>,
    NativeBuilder: VmBuilder<E, VmConfig = NativeConfig>,
    <NativeConfig as VmExecutionConfig<F>>::Executor:
        PreflightExecutor<F, <NativeBuilder as VmBuilder<E>>::RecordArena>,
{
    let mut internal_node_idx = -1;
    let mut internal_node_height = 0;
    let mut all_internal_proofs = Vec::new();
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
        .in_scope(|| {
            internal_inputs
                .into_iter()
                .map(|input| {
                    internal_node_idx += 1;
                    info_span!("single_internal_agg", idx = internal_node_idx).in_scope(|| {
                        SingleSegmentVmProver::prove(
                            &mut agg_prover.internal_prover,
                            input.write(),
                            NATIVE_MAX_TRACE_HEIGHTS,
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()
        })?;

        // Store internal proofs for fixtures
        all_internal_proofs.extend(proofs.clone());
        internal_node_height += 1;
    }

    let proof = proofs.pop().unwrap();
    let vm_stark_proof = VmStarkProof {
        inner: proof,
        user_public_values: public_values,
    };

    Ok((vm_stark_proof, all_internal_proofs))
}

fn main() -> Result<()> {
    // Set up logging
    fmt::fmt().with_env_filter(EnvFilter::new("info")).init();

    let program_dir = get_programs_dir().join(PROGRAM);

    tracing::info!("Loading VM config");
    let config_path = program_dir.join("openvm.toml");
    let config_content = fs::read_to_string(&config_path)?;
    let vm_config = SdkVmConfig::from_toml(&config_content)?.app_vm_config;

    tracing::info!("Preparing ELF");
    let elf_path = get_elf_path(&program_dir);
    let elf = read_elf_file(&elf_path)?;

    // Create app config with default parameters
    let app_config = AppConfig::new(AppFriParams::default().fri_params, vm_config);

    let sdk = Sdk::new(app_config.clone())?;
    let exe = sdk.convert_to_exe(elf)?;

    tracing::info!("Generating app proof");
    let app_proof = sdk.app_prover(exe)?.prove(StdIn::default())?;

    tracing::info!("Getting keys");
    let app_pk = sdk.app_pk();
    let agg_pk = sdk.agg_pk();

    tracing::info!("Creating aggregation provers");
    let native_builder = sdk.native_builder().clone();
    let leaf_verifier_exe = app_pk.leaf_committed_exe.exe.clone();

    let tree_config = AggregationTreeConfig::default();
    let mut agg_prover = AggStarkProver::<BabyBearPoseidon2Engine, _>::new(
        native_builder.clone(),
        agg_pk,
        leaf_verifier_exe,
        tree_config,
    )?;

    tracing::info!("Generating leaf proofs");
    let leaf_proofs = agg_prover.generate_leaf_proofs(&app_proof)?;

    tracing::info!("Generating internal proofs");
    let public_values = app_proof.user_public_values.public_values.clone();
    let (final_proof, internal_proofs) = aggregate_leaf_proofs_and_collect_internals(
        &mut agg_prover,
        leaf_proofs.clone(),
        public_values,
    )?;

    tracing::info!("Saving keys and proofs to files");
    // Create fixtures directory if it doesn't exist
    let fixtures_dir = get_fixtures_dir();
    fs::create_dir_all(&fixtures_dir)?;

    // Serialize and write to files in fixtures directory
    let leaf_exe_bytes = bitcode::serialize(&app_pk.leaf_committed_exe.exe)?;
    fs::write(
        fixtures_dir.join(format!("{}.leaf.exe", PROGRAM)),
        leaf_exe_bytes,
    )?;

    let leaf_pk_bytes = bitcode::serialize(&agg_pk.leaf_vm_pk)?;
    fs::write(
        fixtures_dir.join(format!("{}.leaf.pk", PROGRAM)),
        leaf_pk_bytes,
    )?;

    let internal_pk_bytes = bitcode::serialize(&agg_pk.internal_vm_pk)?;
    fs::write(
        fixtures_dir.join(format!("{}.internal.pk", PROGRAM)),
        internal_pk_bytes,
    )?;

    let app_proof_bytes = bitcode::serialize(&app_proof)?;
    fs::write(
        fixtures_dir.join(format!("{}.app.proof", PROGRAM)),
        app_proof_bytes,
    )?;

    // Save leaf proofs
    for (i, leaf_proof) in leaf_proofs.iter().enumerate() {
        let leaf_proof_bytes = bitcode::serialize(leaf_proof)?;
        fs::write(
            fixtures_dir.join(format!("{}.leaf.{}.proof", PROGRAM, i)),
            leaf_proof_bytes,
        )?;
    }

    // Save internal proofs
    for (i, internal_proof) in internal_proofs.iter().enumerate() {
        let internal_proof_bytes = bitcode::serialize(internal_proof)?;
        fs::write(
            fixtures_dir.join(format!("{}.internal.{}.proof", PROGRAM, i)),
            internal_proof_bytes,
        )?;
    }

    // Save final aggregated proof
    let final_proof_bytes = final_proof.encode_to_vec()?;
    fs::write(
        fixtures_dir.join(format!("{}.final.proof", PROGRAM)),
        final_proof_bytes,
    )?;

    tracing::info!(
        "Generated and saved {name} fixtures: leaf.exe, leaf.pk, internal.pk, app.proof, {} leaf proofs, {} internal proofs, and final.proof",
        leaf_proofs.len(),
        internal_proofs.len(),
        name = PROGRAM
    );

    Ok(())
}
