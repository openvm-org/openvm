use std::{borrow::Borrow, sync::Arc};

use eyre::Result;
use openvm_circuit::{
    arch::{
        hasher::poseidon2::vm_poseidon2_hasher, instructions::exe::VmExe, ContinuationVmProver,
        VirtualMachine, VmInstance,
    },
    system::{
        memory::merkle::public_values::UserPublicValuesProof, program::trace::compute_exe_commit,
    },
    utils::test_utils::test_system_config,
};
use openvm_continuations::{
    circuit::{deferral::DeferralCircuitPvs, utils::vk_commit_components},
    prover::{ChildVkKind, InnerGpuProver as InnerProver},
    utils::poseidon2_input_to_digests,
    SC,
};
use openvm_cuda_backend::{BabyBearPoseidon2GpuEngine, GpuBackend};
use openvm_recursion_circuit::utils::poseidon2_hash_slice;
use openvm_rv32im_circuit::{Rv32IConfig, Rv32ImBuilder, Rv32ImConfig};
use openvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use openvm_stark_backend::{
    keygen::types::MultiStarkVerifyingKey, proof::Proof, prover::CommittedTraceData,
    verifier::verify, StarkEngine, SystemParams, TranscriptHistory,
};
use openvm_stark_sdk::{
    config::{
        app_params_with_100_bits_security,
        baby_bear_poseidon2::{
            default_duplex_sponge_recorder, poseidon2_compress_with_capacity, DIGEST_SIZE, F,
        },
        internal_params_with_100_bits_security, leaf_params_with_100_bits_security,
        root_params_with_100_bits_security,
    },
    utils::setup_tracing_with_log_level,
};
use openvm_transpiler::{
    elf::Elf, openvm_platform::memory::MEM_SIZE, transpiler::Transpiler, FromElf,
};
use openvm_verify_stark_host::pvs::{VerifierBasePvs, VmPvs, VERIFIER_PVS_AIR_ID, VM_PVS_AIR_ID};
use p3_field::PrimeCharacteristicRing;
use test_case::test_case;
use tracing::Level;

type Engine = BabyBearPoseidon2GpuEngine;
type PB = GpuBackend;

const LOG_MAX_TRACE_HEIGHT: usize = 20;
const DEFAULT_MAX_NUM_PROOFS: usize = 4;

fn app_system_params() -> SystemParams {
    app_params_with_100_bits_security(21)
}

fn leaf_system_params() -> SystemParams {
    leaf_params_with_100_bits_security()
}

fn internal_system_params() -> SystemParams {
    internal_params_with_100_bits_security()
}

fn root_system_params() -> SystemParams {
    root_params_with_100_bits_security()
}

fn test_rv32im_config() -> Rv32ImConfig {
    Rv32ImConfig {
        rv32i: Rv32IConfig {
            system: test_system_config().with_max_segment_len(1 << LOG_MAX_TRACE_HEIGHT),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[allow(clippy::type_complexity)]
fn run_leaf_aggregation(
    log_fib_input: usize,
) -> Result<(
    Arc<MultiStarkVerifyingKey<SC>>,
    Proof<SC>,
    UserPublicValuesProof<DIGEST_SIZE, F>,
)> {
    let config = test_rv32im_config();
    let elf = Elf::decode(
        include_bytes!("../../../../crates/continuations/programs/examples/fibonacci.elf"),
        MEM_SIZE as u32,
    )?;
    let exe = VmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension),
    )?;
    let input = (1u64 << log_fib_input)
        .to_le_bytes()
        .map(F::from_u8)
        .to_vec();

    let engine = Engine::new(app_system_params());
    let (vm, app_pk) = VirtualMachine::new_with_keygen(engine, Rv32ImBuilder, config)?;
    let cached_program_trace = vm.commit_program_on_device(&exe.program);
    let mut instance = VmInstance::new(vm, exe.into(), cached_program_trace)?;
    let app_proof = instance.prove(vec![input])?;

    let leaf_prover = InnerProver::<DEFAULT_MAX_NUM_PROOFS>::new::<Engine>(
        Arc::new(app_pk.get_vk()),
        leaf_system_params(),
        false,
        None,
    );
    let leaf_proof =
        leaf_prover.agg_prove_no_def::<Engine>(&app_proof.per_segment, ChildVkKind::App)?;

    let leaf_vk = leaf_prover.get_vk();
    let engine = Engine::new(leaf_vk.inner.params.clone());
    engine.verify(&leaf_vk, &leaf_proof)?;
    Ok((leaf_vk, leaf_proof, app_proof.user_public_values))
}

#[allow(clippy::type_complexity)]
fn run_full_aggregation(
    log_fib_input: usize,
    extra_recursive_layers: usize,
) -> Result<(
    Arc<MultiStarkVerifyingKey<SC>>,
    CommittedTraceData<PB>,
    Proof<SC>,
    UserPublicValuesProof<DIGEST_SIZE, F>,
)> {
    let (leaf_vk, leaf_proof, user_pvs_proof) = run_leaf_aggregation(log_fib_input)?;

    let internal_for_leaf_prover = InnerProver::<DEFAULT_MAX_NUM_PROOFS>::new::<Engine>(
        leaf_vk,
        internal_system_params(),
        false,
        None,
    );
    let internal_for_leaf_proof = internal_for_leaf_prover
        .agg_prove_no_def::<Engine>(&[leaf_proof], ChildVkKind::Standard)?;

    let internal_recursive_prover = InnerProver::<DEFAULT_MAX_NUM_PROOFS>::new::<Engine>(
        internal_for_leaf_prover.get_vk(),
        internal_system_params(),
        true,
        None,
    );
    let mut internal_recursive_proof = internal_recursive_prover
        .agg_prove_no_def::<Engine>(&[internal_for_leaf_proof], ChildVkKind::Standard)?;

    for _ in 0..extra_recursive_layers {
        internal_recursive_proof = internal_recursive_prover
            .agg_prove_no_def::<Engine>(&[internal_recursive_proof], ChildVkKind::RecursiveSelf)?;
    }

    Ok((
        internal_recursive_prover.get_vk(),
        internal_recursive_prover.get_self_vk_pcs_data().unwrap(),
        internal_recursive_proof,
        user_pvs_proof,
    ))
}

#[test_case(0 ; "internal_recursive_dag_commit not set")]
#[test_case(1 ; "internal_recursive_dag_commit set")]
fn test_deferral_verify_prover(child_extra_recursive_layers: usize) -> Result<()> {
    setup_tracing_with_log_level(Level::INFO);
    let (
        internal_recursive_vk,
        internal_recursive_pcs_data,
        internal_recursive_proof,
        user_pvs_proof,
    ) = run_full_aggregation(10, child_extra_recursive_layers)?;

    let system_config = test_rv32im_config().rv32i.system;
    let deferred_verify_prover = crate::prover::DeferredVerifyDefaultProver::new::<Engine>(
        internal_recursive_vk.clone(),
        internal_recursive_pcs_data,
        root_system_params(),
        system_config.memory_config.memory_dimensions(),
        system_config.num_public_values,
        None,
    );
    let def_proof = deferred_verify_prover
        .prove_no_def::<Engine>(internal_recursive_proof.clone(), &user_pvs_proof)?;

    let vk = deferred_verify_prover.get_vk();
    let engine = Engine::new(vk.inner.params.clone());
    engine.verify(&vk, &def_proof)?;

    let mut ts = default_duplex_sponge_recorder();
    let config = SC::default_from_params(internal_recursive_vk.inner.params.clone());
    verify(
        &config,
        internal_recursive_vk.as_ref(),
        &internal_recursive_proof,
        &mut ts,
    )?;
    let final_ts_state = *ts.into_log().perm_results().last().unwrap();
    let (left, right) = poseidon2_input_to_digests(final_ts_state);
    let expected_input_commit = poseidon2_compress_with_capacity(left, right).0;

    let (base_pvs_slice, _) = internal_recursive_proof.public_values[VERIFIER_PVS_AIR_ID]
        .as_slice()
        .split_at(VerifierBasePvs::<u8>::width());
    let verifier_pvs: &VerifierBasePvs<F> = base_pvs_slice.borrow();
    let vm_pvs: &VmPvs<F> = internal_recursive_proof.public_values[VM_PVS_AIR_ID]
        .as_slice()
        .borrow();

    let app_exe_commit = compute_exe_commit(
        &vm_poseidon2_hasher::<F>(),
        &vm_pvs.program_commit,
        &vm_pvs.initial_root,
        vm_pvs.initial_pc,
    );
    let app_vk_commit =
        poseidon2_hash_slice(&vk_commit_components(verifier_pvs).into_flattened()).0;
    let expected_output_commit = crate::output::generate_proving_ctx(
        app_exe_commit,
        app_vk_commit,
        user_pvs_proof.public_values,
    )
    .output_commit;

    let def_pvs: &DeferralCircuitPvs<F> = def_proof.public_values[VERIFIER_PVS_AIR_ID]
        .as_slice()
        .borrow();
    assert_eq!(expected_input_commit, def_pvs.input_commit);
    assert_eq!(expected_output_commit, def_pvs.output_commit);

    Ok(())
}
