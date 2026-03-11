use std::sync::Arc;

use eyre::Result;
use itertools::Itertools;
use openvm_circuit::{
    arch::{
        instructions::exe::VmExe, ContinuationVmProver, VirtualMachine, VmCircuitConfig, VmInstance,
    },
    system::memory::merkle::public_values::UserPublicValuesProof,
    utils::test_utils::test_system_config,
};
use openvm_rv32im_circuit::{Rv32IConfig, Rv32ImBuilder, Rv32ImConfig};
use openvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use openvm_stark_backend::{
    keygen::types::MultiStarkVerifyingKey, proof::Proof, StarkEngine, SystemParams,
};
use openvm_stark_sdk::{
    config::{
        app_params_with_100_bits_security,
        baby_bear_poseidon2::{DIGEST_SIZE, F},
        internal_params_with_100_bits_security, leaf_params_with_100_bits_security,
    },
    utils::setup_tracing_with_log_level,
};
use openvm_transpiler::{
    elf::Elf, openvm_platform::memory::MEM_SIZE, transpiler::Transpiler, FromElf,
};
use p3_field::PrimeCharacteristicRing;
use test_case::test_case;
use tracing::Level;

use crate::{prover::ChildVkKind, SC};

#[cfg(feature = "cuda")]
mod e2e;
#[cfg(feature = "cuda")]
mod verify;

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use std::borrow::Borrow;
        use crate::prover::InnerGpuProver as InnerProver;
        use crate::prover::CompressionGpuProver as CompressionProver;
        use crate::prover::RootCpuProver as RootProver;
        use crate::prover::{
            DeferralInnerGpuProver as DeferralInnerProver,
            DeferralHookGpuProver as DeferralHookProver,
            DeferralVerifyGpuProver as DeferredVerifyProver,
        };
        use crate::circuit::{
            deferral::{
                verify::output::expected_output_commit,
                DeferralAggregationPvs,
                DeferralCircuitPvs,
                DEF_AGG_PVS_AIR_ID,
            },
            root::RootVerifierPvs,
        };
        use recursion_circuit::utils::poseidon2_hash_slice_with_states;
        use openvm_cuda_backend::{BabyBearPoseidon2GpuEngine, GpuBackend};
        use openvm_stark_backend::{prover::CommittedTraceData, verifier::verify, TranscriptHistory};
        use openvm_stark_sdk::config::{
            baby_bear_poseidon2::{poseidon2_compress_with_capacity, default_duplex_sponge_recorder},
            baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2CpuEngine,
        };
        use verify_stark::pvs::{VERIFIER_PVS_AIR_ID, DeferralPvs, VerifierBasePvs};
        use crate::prover::DeferralChildVkKind;
        use crate::utils::{poseidon2_input_to_digests, zero_hash};
        type RootEngine = BabyBearBn254Poseidon2CpuEngine;
        type Engine = BabyBearPoseidon2GpuEngine;
        type PB = GpuBackend;
    } else {
        use crate::prover::InnerCpuProver as InnerProver;
        use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2CpuEngine, DuplexSponge};
        type Engine = BabyBearPoseidon2CpuEngine<DuplexSponge>;
    }
}

const LOG_MAX_TRACE_HEIGHT: usize = 20;
const DEFAULT_MAX_NUM_PROOFS: usize = 4;

pub(in crate::tests) fn app_system_params() -> SystemParams {
    app_params_with_100_bits_security(21)
}

pub(in crate::tests) fn leaf_system_params() -> SystemParams {
    leaf_params_with_100_bits_security()
}

pub(in crate::tests) fn internal_system_params() -> SystemParams {
    internal_params_with_100_bits_security()
}

#[cfg(feature = "cuda")]
pub(in crate::tests) fn compression_system_params() -> SystemParams {
    use openvm_stark_sdk::config::compression_params_with_100_bits_security;

    compression_params_with_100_bits_security()
}

#[cfg(feature = "cuda")]
pub(in crate::tests) fn root_system_params() -> SystemParams {
    use openvm_stark_sdk::config::root_params_with_100_bits_security;

    root_params_with_100_bits_security()
}

pub(in crate::tests) fn test_rv32im_config() -> Rv32ImConfig {
    Rv32ImConfig {
        rv32i: Rv32IConfig {
            system: test_system_config().with_max_segment_len(1 << LOG_MAX_TRACE_HEIGHT),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[allow(clippy::type_complexity)]
pub(in crate::tests) fn run_leaf_aggregation(
    log_fib_input: usize,
) -> Result<(
    Arc<MultiStarkVerifyingKey<SC>>,
    Proof<SC>,
    UserPublicValuesProof<DIGEST_SIZE, F>,
)> {
    let config = test_rv32im_config();
    let elf = Elf::decode(
        include_bytes!("../../programs/examples/fibonacci.elf"),
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

// Feature-gated because full aggregation is too slow without CUDA. Many tests below
// (including all that use run_full_aggregation) are similarly gated.
#[cfg(feature = "cuda")]
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

    for _internal_recursive_layer in 0..extra_recursive_layers {
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

#[test]
fn test_single_segment_leaf_aggregation() -> Result<()> {
    setup_tracing_with_log_level(Level::INFO);
    run_leaf_aggregation(5)?;
    Ok(())
}

#[test]
fn test_two_segments_leaf_aggregation() -> Result<()> {
    setup_tracing_with_log_level(Level::INFO);
    run_leaf_aggregation(17)?;
    Ok(())
}

#[test_case(false ; "def_hook_commit not set")]
#[test_case(true ; "def_hook_commit set")]
fn test_internal_recursive_vk_stabilization(def_hook_commit_set: bool) -> Result<()> {
    setup_tracing_with_log_level(Level::INFO);
    let config = test_rv32im_config();

    let engine = Engine::new(app_system_params());
    let (_, app_vk) = engine.keygen(&config.create_airs()?.into_airs().collect_vec());
    let def_hook_commit = def_hook_commit_set.then_some([F::ZERO; DIGEST_SIZE]);

    const MAX_LEAF_NUM_PROOFS: usize = 3;
    let leaf_prover = InnerProver::<MAX_LEAF_NUM_PROOFS>::new::<Engine>(
        Arc::new(app_vk),
        leaf_system_params(),
        false,
        def_hook_commit,
    );
    let internal_0_prover = InnerProver::<DEFAULT_MAX_NUM_PROOFS>::new::<Engine>(
        leaf_prover.get_vk(),
        internal_system_params(),
        false,
        def_hook_commit,
    );
    let internal_1_prover = InnerProver::<DEFAULT_MAX_NUM_PROOFS>::new::<Engine>(
        internal_0_prover.get_vk(),
        internal_system_params(),
        false,
        def_hook_commit,
    );

    // The internal vk should stabilize at the second internal layer
    let test_prover = InnerProver::<DEFAULT_MAX_NUM_PROOFS>::new::<Engine>(
        internal_1_prover.get_vk(),
        internal_system_params(),
        true,
        def_hook_commit,
    );
    assert_eq!(
        test_prover.get_dag_commit(false),
        test_prover.get_dag_commit(true)
    );
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_internal_recursive_deep_layers() -> Result<()> {
    // This test is too slow to run on CPU
    setup_tracing_with_log_level(Level::INFO);
    let (internal_recursive_vk, _, internal_recursive_proof, _) = run_full_aggregation(10, 3)?;
    let engine = Engine::new(internal_recursive_vk.inner.params.clone());
    engine.verify(&internal_recursive_vk, &internal_recursive_proof)?;
    Ok(())
}

#[cfg(feature = "cuda")]
#[test_case(0 ; "internal_recursive_dag_commit not set")]
#[test_case(1 ; "internal_recursive_dag_commit set")]
fn test_compression_prover(extra_recursive_layers: usize) -> Result<()> {
    setup_tracing_with_log_level(Level::INFO);
    let (internal_recursive_vk, internal_recursive_pcs_data, internal_recursive_proof, _) =
        run_full_aggregation(10, extra_recursive_layers)?;

    let compression_prover = CompressionProver::new::<Engine>(
        internal_recursive_vk,
        internal_recursive_pcs_data,
        compression_system_params(),
        None,
    );
    let compression_proof =
        compression_prover.compress_prove_no_def::<Engine>(internal_recursive_proof)?;

    let vk = compression_prover.get_vk();
    let engine = Engine::new(vk.inner.params.clone());
    engine.verify(&vk, &compression_proof)?;
    Ok(())
}

#[cfg(feature = "cuda")]
#[test_case(0 ; "internal_recursive_dag_commit not set")]
#[test_case(1 ; "internal_recursive_dag_commit set")]
fn test_root_prover(extra_recursive_layers: usize) -> Result<()> {
    setup_tracing_with_log_level(Level::INFO);
    let (
        internal_recursive_vk,
        internal_recursive_pcs_data,
        internal_recursive_proof,
        user_pvs_proof,
    ) = run_full_aggregation(10, extra_recursive_layers)?;

    let system_config = test_rv32im_config().rv32i.system;

    let root_prover = RootProver::new::<RootEngine>(
        internal_recursive_vk,
        internal_recursive_pcs_data.commitment.into(),
        root_system_params(),
        system_config.memory_config.memory_dimensions(),
        system_config.num_public_values,
        None,
        None,
    );
    let ctx = root_prover.generate_proving_ctx(internal_recursive_proof, &user_pvs_proof);
    let root_proof = root_prover.root_prove_from_ctx::<RootEngine>(ctx.unwrap())?;

    let vk = root_prover.get_vk();
    let engine = RootEngine::new(vk.inner.params.clone());
    engine.verify(&vk, &root_proof)?;
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_root_prover_trace_heights() -> Result<()> {
    setup_tracing_with_log_level(Level::INFO);
    let (
        internal_recursive_vk,
        internal_recursive_pcs_data,
        internal_recursive_proof,
        user_pvs_proof,
    ) = run_full_aggregation(10, 1)?;

    let system_config = test_rv32im_config().rv32i.system;

    let root_base_prover = RootProver::new::<RootEngine>(
        internal_recursive_vk.clone(),
        internal_recursive_pcs_data.commitment.into(),
        root_system_params(),
        system_config.memory_config.memory_dimensions(),
        system_config.num_public_values,
        None,
        None,
    );
    let ctx = root_base_prover
        .generate_proving_ctx(internal_recursive_proof.clone(), &user_pvs_proof)
        .unwrap();
    let mut trace_heights = ctx
        .per_trace
        .iter()
        .map(|(_, air_ctx)| air_ctx.height())
        .collect_vec();

    const AIR_MODIFIED_HEIGHT_IDX: usize = 4;
    trace_heights[AIR_MODIFIED_HEIGHT_IDX] *= 2;

    let root_prover = RootProver::new::<RootEngine>(
        internal_recursive_vk,
        internal_recursive_pcs_data.commitment.into(),
        root_system_params(),
        system_config.memory_config.memory_dimensions(),
        system_config.num_public_values,
        None,
        Some(trace_heights.clone()),
    );
    let ctx = root_prover
        .generate_proving_ctx(internal_recursive_proof, &user_pvs_proof)
        .unwrap();

    for ((air_idx, air_ctx), expected_height) in ctx.per_trace.iter().zip(trace_heights) {
        assert_eq!(air_ctx.height(), expected_height, "air_idx {air_idx}");
    }
    let root_proof = root_prover.root_prove_from_ctx::<RootEngine>(ctx)?;

    let vk = root_prover.get_vk();
    let engine = RootEngine::new(vk.inner.params.clone());
    engine.verify(&vk, &root_proof)?;
    Ok(())
}

///////////////////////////////////////////////////////////////////////////////
// DEFERRAL TESTS
///////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "cuda")]
fn generate_single_def_proof(
    child_extra_recursive_layers: usize,
) -> Result<(Arc<MultiStarkVerifyingKey<SC>>, Proof<SC>)> {
    let (
        internal_recursive_vk,
        internal_recursive_pcs_data,
        internal_recursive_proof,
        user_pvs_proof,
    ) = run_full_aggregation(10, child_extra_recursive_layers)?;
    let system_config = test_rv32im_config().rv32i.system;
    let deferred_verify_prover = DeferredVerifyProver::new::<Engine>(
        internal_recursive_vk,
        internal_recursive_pcs_data,
        root_system_params(),
        system_config.memory_config.memory_dimensions(),
        system_config.num_public_values,
        None,
    );
    let def_proof =
        deferred_verify_prover.prove_no_def::<Engine>(internal_recursive_proof, &user_pvs_proof)?;
    Ok((deferred_verify_prover.get_vk(), def_proof))
}

#[cfg(feature = "cuda")]
fn aggregate_deferral_layer(
    prover: &DeferralInnerProver<2>,
    proofs: &[Proof<SC>],
    child_is_def: bool,
    child_merkle_depth: usize,
) -> Result<Vec<Proof<SC>>> {
    assert!(!proofs.is_empty(), "proof layer must be non-empty");
    let mut next = Vec::with_capacity(proofs.len().div_ceil(2));
    let layer_merkle_depth = if proofs.len() == 1 {
        None
    } else {
        Some(child_merkle_depth)
    };
    for chunk in proofs.chunks(2) {
        let child_vk_kind = if child_is_def {
            DeferralChildVkKind::DeferralAggregation
        } else {
            DeferralChildVkKind::DeferralCircuit
        };
        let proof = if chunk.len() == 2 {
            prover.agg_prove::<Engine>(
                &[chunk[0].clone(), chunk[1].clone()],
                child_vk_kind,
                layer_merkle_depth,
            )?
        } else {
            prover.agg_prove::<Engine>(&[chunk[0].clone()], child_vk_kind, layer_merkle_depth)?
        };
        next.push(proof);
    }
    Ok(next)
}

#[cfg(feature = "cuda")]
pub(in crate::tests) fn generate_deferral_internal_recursive_proof_from_copies(
    deferral_vk: Arc<MultiStarkVerifyingKey<SC>>,
    def_proof: Proof<SC>,
    num_copies: usize,
) -> Result<(Arc<MultiStarkVerifyingKey<SC>>, Proof<SC>)> {
    assert!(num_copies > 0, "num_copies must be non-zero");

    let mut current_proofs = vec![def_proof.clone(); num_copies];
    let mut child_merkle_depth = 0usize;

    let leaf_prover =
        DeferralInnerProver::<2>::new::<Engine>(deferral_vk, leaf_system_params(), false);
    current_proofs =
        aggregate_deferral_layer(&leaf_prover, &current_proofs, false, child_merkle_depth)?;
    child_merkle_depth += 1;

    let internal_for_leaf_prover = DeferralInnerProver::<2>::new::<Engine>(
        leaf_prover.get_vk(),
        internal_system_params(),
        false,
    );
    current_proofs = aggregate_deferral_layer(
        &internal_for_leaf_prover,
        &current_proofs,
        true,
        child_merkle_depth,
    )?;
    child_merkle_depth += 1;

    let child_vk = internal_for_leaf_prover.get_vk();
    let internal_recursive_prover =
        DeferralInnerProver::<2>::new::<Engine>(child_vk, internal_system_params(), true);
    loop {
        current_proofs = aggregate_deferral_layer(
            &internal_recursive_prover,
            &current_proofs,
            true,
            child_merkle_depth,
        )?;
        child_merkle_depth += 1;

        if current_proofs.len() == 1 {
            let wrapped = internal_recursive_prover.agg_prove::<Engine>(
                &[current_proofs[0].clone()],
                DeferralChildVkKind::RecursiveSelf,
                None,
            )?;
            return Ok((internal_recursive_prover.get_vk(), wrapped));
        }
    }
}

#[cfg(feature = "cuda")]
fn expected_deferral_leaf_merkle_commit(def_proof: &Proof<SC>) -> [F; DIGEST_SIZE] {
    let (folded_input_commit, output_commit) = expected_deferral_leaf_io_commit(def_proof);
    poseidon2_compress_with_capacity(folded_input_commit, output_commit).0
}

#[cfg(feature = "cuda")]
pub(in crate::tests) fn expected_deferral_leaf_io_commit(
    def_proof: &Proof<SC>,
) -> ([F; DIGEST_SIZE], [F; DIGEST_SIZE]) {
    let def_pvs: &DeferralCircuitPvs<F> = def_proof.public_values[VERIFIER_PVS_AIR_ID]
        .as_slice()
        .borrow();
    let folded_input_commit = def_proof
        .trace_vdata
        .iter()
        .flatten()
        .flat_map(|vdata| vdata.cached_commitments.iter().copied())
        .fold(def_pvs.input_commit, |acc, cached_commit| {
            poseidon2_compress_with_capacity(acc, cached_commit).0
        });
    (folded_input_commit, def_pvs.output_commit)
}

#[cfg(feature = "cuda")]
fn expected_deferral_inner_merkle_commit_from_copies(
    num_copies: usize,
    leaf_merkle_commit: [F; DIGEST_SIZE],
) -> [F; DIGEST_SIZE] {
    assert!(num_copies > 0, "num_copies must be non-zero");

    let mut commits = vec![leaf_merkle_commit; num_copies];
    let mut child_merkle_depth = 0usize;

    while commits.len() > 1 {
        commits = commits
            .chunks(2)
            .map(|chunk| {
                let right = if chunk.len() == 2 {
                    chunk[1]
                } else {
                    zero_hash(child_merkle_depth + 1)
                };
                poseidon2_compress_with_capacity(chunk[0], right).0
            })
            .collect();
        child_merkle_depth += 1;
    }
    commits[0]
}

#[cfg(feature = "cuda")]
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

    // Compute def_proof
    let deferred_verify_prover = DeferredVerifyProver::new::<Engine>(
        internal_recursive_vk.clone(),
        internal_recursive_pcs_data.clone(),
        root_system_params(),
        system_config.memory_config.memory_dimensions(),
        system_config.num_public_values,
        None,
    );
    let def_proof = deferred_verify_prover
        .prove_no_def::<Engine>(internal_recursive_proof.clone(), &user_pvs_proof)?;

    // Verify that def_proof is valid
    let vk = deferred_verify_prover.get_vk();
    let engine = Engine::new(vk.inner.params.clone());
    engine.verify(&vk, &def_proof)?;

    // Get the final transcript state of internal_recursive_proof
    let mut ts = default_duplex_sponge_recorder();
    let config = SC::default_from_params(internal_recursive_vk.inner.params.clone());
    verify(
        &config,
        internal_recursive_vk.as_ref(),
        &internal_recursive_proof,
        &mut ts,
    )?;
    let ts_log = ts.into_log();
    let expected_final_ts_state = ts_log.perm_results().last().unwrap();

    // Generate a root_proof to compare the pvs of def_proof against
    let root_prover = RootProver::new::<RootEngine>(
        internal_recursive_vk,
        internal_recursive_pcs_data.commitment.into(),
        root_system_params(),
        system_config.memory_config.memory_dimensions(),
        system_config.num_public_values,
        None,
        None,
    );
    let ctx = root_prover.generate_proving_ctx(internal_recursive_proof, &user_pvs_proof);
    let root_proof = root_prover.root_prove_from_ctx::<RootEngine>(ctx.unwrap())?;

    // Assert the correctness of the def_proof public values
    let root_pvs: &RootVerifierPvs<F> = root_proof.public_values[VERIFIER_PVS_AIR_ID]
        .as_slice()
        .borrow();
    // RootCircuit AIR layout is:
    // 0: RootVerifierPvsAir, 1: UserPvsCommitAir, 2: UserPvsInMemoryAir, 3..: verifier subcircuit.
    const ROOT_USER_PVS_COMMIT_AIR_ID: usize = 1;
    let user_pvs = root_proof.public_values[ROOT_USER_PVS_COMMIT_AIR_ID].clone();

    let (left, right) = poseidon2_input_to_digests(*expected_final_ts_state);
    let expected_input_commit = poseidon2_compress_with_capacity(left, right).0;
    let expected_output_commit =
        expected_output_commit(root_pvs.app_exe_commit, root_pvs.app_vk_commit, user_pvs);

    let def_pvs: &DeferralCircuitPvs<F> = def_proof.public_values[VERIFIER_PVS_AIR_ID]
        .as_slice()
        .borrow();
    assert_eq!(expected_input_commit, def_pvs.input_commit);
    assert_eq!(expected_output_commit, def_pvs.output_commit);

    Ok(())
}

#[cfg(feature = "cuda")]
#[test_case(0, 1 ; "internal_recursive_dag_commit not set + one child")]
#[test_case(1, 1 ; "internal_recursive_dag_commit set + one child")]
#[test_case(0, 2 ; "internal_recursive_dag_commit not set + two children")]
#[test_case(1, 2 ; "internal_recursive_dag_commit set + two children")]
fn test_deferral_leaf_prover(
    child_extra_recursive_layers: usize,
    num_children: usize,
) -> Result<()> {
    setup_tracing_with_log_level(Level::INFO);
    let (deferral_vk, def_proof) = generate_single_def_proof(child_extra_recursive_layers)?;

    let deferral_inner_prover =
        DeferralInnerProver::<2>::new::<Engine>(deferral_vk, leaf_system_params(), false);
    let wrapped_proof = deferral_inner_prover.agg_prove::<Engine>(
        &vec![def_proof.clone(); num_children],
        DeferralChildVkKind::DeferralCircuit,
        if num_children == 1 { None } else { Some(0) },
    )?;

    // Verify wrapped proof.
    let vk = deferral_inner_prover.get_vk();
    let engine = Engine::new(vk.inner.params.clone());
    engine.verify(&vk, &wrapped_proof)?;

    // Assert DeferralAggregationPvs consistency for this aggregation size.
    let leaf_merkle_commit = expected_deferral_leaf_merkle_commit(&def_proof);
    let expected_merkle_commit =
        expected_deferral_inner_merkle_commit_from_copies(num_children, leaf_merkle_commit);

    let wrapped_pvs: &DeferralAggregationPvs<F> = wrapped_proof.public_values[DEF_AGG_PVS_AIR_ID]
        .as_slice()
        .borrow();
    assert_eq!(expected_merkle_commit, wrapped_pvs.merkle_commit);

    Ok(())
}

#[cfg(feature = "cuda")]
#[test_case(1 ; "single def_circuit proof")]
#[test_case(4 ; "full aggregation tree")]
#[test_case(5 ; "partially empty aggregation tree")]
fn test_deferral_aggregation(num_children: usize) -> Result<()> {
    setup_tracing_with_log_level(Level::INFO);
    let (deferral_vk, def_proof) = generate_single_def_proof(0)?;
    let (vk, final_proof) = generate_deferral_internal_recursive_proof_from_copies(
        deferral_vk,
        def_proof.clone(),
        num_children,
    )?;

    let engine = Engine::new(vk.inner.params.clone());
    engine.verify(&vk, &final_proof)?;

    let leaf_merkle_commit = expected_deferral_leaf_merkle_commit(&def_proof);
    let expected_root_merkle =
        expected_deferral_inner_merkle_commit_from_copies(num_children, leaf_merkle_commit);

    let wrapped_pvs: &DeferralAggregationPvs<F> = final_proof.public_values[DEF_AGG_PVS_AIR_ID]
        .as_slice()
        .borrow();
    assert_eq!(expected_root_merkle, wrapped_pvs.merkle_commit);

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_deferral_internal_recursive_vk_stabilization() -> Result<()> {
    setup_tracing_with_log_level(Level::INFO);
    let (deferral_vk, _) = generate_single_def_proof(0)?;

    let leaf_prover =
        DeferralInnerProver::<2>::new::<Engine>(deferral_vk, leaf_system_params(), false);
    let internal_0_prover = DeferralInnerProver::<2>::new::<Engine>(
        leaf_prover.get_vk(),
        internal_system_params(),
        false,
    );
    let internal_1_prover = DeferralInnerProver::<2>::new::<Engine>(
        internal_0_prover.get_vk(),
        internal_system_params(),
        false,
    );
    let test_prover = DeferralInnerProver::<2>::new::<Engine>(
        internal_1_prover.get_vk(),
        internal_system_params(),
        true,
    );

    assert_eq!(
        test_prover.get_dag_commit(false),
        test_prover.get_dag_commit(true)
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[test_case(1 ; "single def_circuit proof")]
#[test_case(4 ; "full aggregation tree")]
#[test_case(5 ; "partially empty aggregation tree")]
fn test_deferral_hook_prover(num_children: usize) -> Result<()> {
    use crate::circuit::utils::vk_commit_components;

    setup_tracing_with_log_level(Level::INFO);
    let (deferral_vk, def_proof) = generate_single_def_proof(0)?;
    let (deferral_internal_recursive_vk, final_inner_proof) =
        generate_deferral_internal_recursive_proof_from_copies(
            deferral_vk,
            def_proof.clone(),
            num_children,
        )?;
    let (leaf_input_commit, leaf_output_commit) = expected_deferral_leaf_io_commit(&def_proof);

    let leaf_commit = (leaf_input_commit, leaf_output_commit);
    let leaf_children = vec![leaf_commit; num_children];

    let deferral_hook_prover =
        DeferralHookProver::new::<Engine>(deferral_internal_recursive_vk, root_system_params());
    let root_proof =
        deferral_hook_prover.prove::<Engine>(final_inner_proof.clone(), leaf_children)?;

    let root_vk = deferral_hook_prover.get_vk();
    let engine = Engine::new(root_vk.inner.params.clone());
    engine.verify(&root_vk, &root_proof)?;

    let child_verifier_pvs: &VerifierBasePvs<F> =
        final_inner_proof.public_values[0].as_slice().borrow();

    // app_dag_commit is def_dag_commit here
    let def_vk = poseidon2_hash_slice_with_states(
        &vk_commit_components(child_verifier_pvs)
            .into_iter()
            .flatten()
            .collect_vec(),
    )
    .0;

    let mut expected_input_onion = def_vk;
    let mut expected_output_onion = [F::ZERO; DIGEST_SIZE];
    for _ in 0..num_children {
        expected_input_onion =
            poseidon2_compress_with_capacity(expected_input_onion, leaf_input_commit).0;
        expected_output_onion =
            poseidon2_compress_with_capacity(expected_output_onion, leaf_output_commit).0;
    }
    let root_pvs: &DeferralPvs<F> = root_proof.public_values[0].as_slice().borrow();
    let expected_initial_acc_hash = poseidon2_compress_with_capacity(
        poseidon2_compress_with_capacity(def_vk, [F::ZERO; DIGEST_SIZE]).0,
        zero_hash(1),
    )
    .0;
    let expected_final_acc_hash = poseidon2_compress_with_capacity(
        poseidon2_compress_with_capacity(expected_input_onion, [F::ZERO; DIGEST_SIZE]).0,
        poseidon2_compress_with_capacity(expected_output_onion, [F::ZERO; DIGEST_SIZE]).0,
    )
    .0;
    assert_eq!(root_pvs.initial_acc_hash, expected_initial_acc_hash);
    assert_eq!(root_pvs.final_acc_hash, expected_final_acc_hash);

    Ok(())
}
