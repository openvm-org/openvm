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
    interaction::LogUpSecurityParameters, keygen::types::MultiStarkVerifyingKey, proof::Proof,
    StarkEngine, SystemParams, WhirConfig, WhirParams,
};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::{DIGEST_SIZE, F},
    p3_baby_bear::BabyBear,
    utils::setup_tracing_with_log_level,
};
use openvm_transpiler::{
    elf::Elf, openvm_platform::memory::MEM_SIZE, transpiler::Transpiler, FromElf,
};
use p3_field::{PrimeCharacteristicRing, PrimeField32};
#[cfg(feature = "cuda")]
use test_case::test_case;
use tracing::Level;

use crate::{aggregation::ChildVkKind, SC};

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use crate::aggregation::NonRootGpuProver as NonRootProver;
        use crate::aggregation::CompressionGpuProver as CompressionProver;
        use crate::aggregation::RootGpuProver as RootProver;
        use openvm_cuda_backend::{BabyBearPoseidon2GpuEngine, GpuBackend};
        use openvm_stark_backend::prover::CommittedTraceData;
        type Engine = BabyBearPoseidon2GpuEngine;
        type PB = GpuBackend;
    } else {
        use crate::aggregation::NonRootCpuProver as NonRootProver;
        use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2CpuEngine, DuplexSponge};
        type Engine = BabyBearPoseidon2CpuEngine<DuplexSponge>;
    }
}

const LOG_MAX_TRACE_HEIGHT: usize = 20;
const DEFAULT_MAX_NUM_PROOFS: usize = 4;

fn app_system_params() -> SystemParams {
    let l_skip = 4;
    let n_stack = 17;
    let log_blowup = 1;
    let whir_params = WhirParams {
        k: 4,
        log_final_poly_len: 10,
        query_phase_pow_bits: 16,
    };
    let whir = WhirConfig::new(log_blowup, l_skip + n_stack, whir_params, 100);
    SystemParams {
        l_skip,
        n_stack,
        log_blowup,
        whir,
        logup: LogUpSecurityParameters {
            max_interaction_count: BabyBear::ORDER_U32,
            log_max_message_length: 7,
            pow_bits: 16,
        },
        max_constraint_degree: 4,
    }
}

fn leaf_system_params() -> SystemParams {
    app_system_params()
}

fn internal_system_params() -> SystemParams {
    app_system_params()
}

#[cfg(feature = "cuda")]
fn compression_system_params() -> SystemParams {
    app_system_params()
}

#[cfg(feature = "cuda")]
fn root_system_params() -> SystemParams {
    app_system_params()
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

fn run_leaf_aggregation(
    log_fib_input: usize,
) -> Result<(
    Arc<MultiStarkVerifyingKey<SC>>,
    Proof<SC>,
    UserPublicValuesProof<DIGEST_SIZE, F>,
)> {
    let config = test_rv32im_config();
    let elf = Elf::decode(
        include_bytes!("../programs/examples/fibonacci.elf"),
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

    let leaf_prover = NonRootProver::<DEFAULT_MAX_NUM_PROOFS>::new::<Engine>(
        Arc::new(app_pk.get_vk()),
        leaf_system_params(),
        false,
    );
    let leaf_proof = leaf_prover.agg_prove::<Engine>(&app_proof.per_segment, ChildVkKind::App)?;

    let leaf_vk = leaf_prover.get_vk();
    let engine = Engine::new(leaf_vk.inner.params.clone());
    engine.verify(&leaf_vk, &leaf_proof)?;
    Ok((leaf_vk, leaf_proof, app_proof.user_public_values))
}

// Feature-gated because full aggregation is too slow without CUDA. Many tests below
// (including all that use run_full_aggregation) are similarly gated.
#[cfg(feature = "cuda")]
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

    let internal_for_leaf_prover = NonRootProver::<DEFAULT_MAX_NUM_PROOFS>::new::<Engine>(
        leaf_vk,
        internal_system_params(),
        false,
    );
    let internal_for_leaf_proof =
        internal_for_leaf_prover.agg_prove::<Engine>(&[leaf_proof], ChildVkKind::Standard)?;

    let internal_recursive_prover = NonRootProver::<DEFAULT_MAX_NUM_PROOFS>::new::<Engine>(
        internal_for_leaf_prover.get_vk(),
        internal_system_params(),
        true,
    );
    let mut internal_recursive_proof = internal_recursive_prover
        .agg_prove::<Engine>(&[internal_for_leaf_proof], ChildVkKind::Standard)?;

    for _internal_recursive_layer in 0..extra_recursive_layers {
        internal_recursive_proof = internal_recursive_prover
            .agg_prove::<Engine>(&[internal_recursive_proof], ChildVkKind::RecursiveSelf)?;
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
#[cfg(feature = "cuda")]
fn test_two_segments_leaf_aggregation() -> Result<()> {
    // This test is too slow to run on CPU
    setup_tracing_with_log_level(Level::INFO);
    run_leaf_aggregation(17)?;
    Ok(())
}

#[test]
fn test_internal_recursive_vk_stabilization() -> Result<()> {
    setup_tracing_with_log_level(Level::INFO);
    let config = test_rv32im_config();

    let engine = Engine::new(app_system_params());
    let (_, app_vk) = engine.keygen(&config.create_airs()?.into_airs().collect_vec());

    let leaf_prover = NonRootProver::<DEFAULT_MAX_NUM_PROOFS>::new::<Engine>(
        Arc::new(app_vk),
        leaf_system_params(),
        false,
    );
    let internal_0_prover = NonRootProver::<DEFAULT_MAX_NUM_PROOFS>::new::<Engine>(
        leaf_prover.get_vk(),
        internal_system_params(),
        false,
    );
    let internal_1_prover = NonRootProver::<DEFAULT_MAX_NUM_PROOFS>::new::<Engine>(
        internal_0_prover.get_vk(),
        internal_system_params(),
        false,
    );

    // The internal vk should stabilize at the second internal layer
    let test_prover = NonRootProver::<DEFAULT_MAX_NUM_PROOFS>::new::<Engine>(
        internal_1_prover.get_vk(),
        internal_system_params(),
        true,
    );
    assert_eq!(
        test_prover.get_cached_commit(false),
        test_prover.get_cached_commit(true)
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

#[test]
#[cfg(feature = "cuda")]
fn test_compression_prover() -> Result<()> {
    setup_tracing_with_log_level(Level::INFO);
    let (internal_recursive_vk, internal_recursive_pcs_data, internal_recursive_proof, _) =
        run_full_aggregation(10, 0)?;

    let compression_prover = CompressionProver::new::<Engine>(
        internal_recursive_vk,
        internal_recursive_pcs_data,
        compression_system_params(),
    );
    let compression_proof =
        compression_prover.compress_prove::<Engine>(internal_recursive_proof)?;

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

    let root_prover = RootProver::new::<Engine>(
        internal_recursive_vk,
        internal_recursive_pcs_data,
        root_system_params(),
        system_config.memory_config.memory_dimensions(),
        system_config.num_public_values,
        None,
    );
    let ctx = root_prover.generate_proving_ctx(internal_recursive_proof, &user_pvs_proof);
    let root_proof = root_prover.root_prove_from_ctx::<Engine>(ctx.unwrap())?;

    let vk = root_prover.get_vk();
    let engine = Engine::new(vk.inner.params.clone());
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

    let root_base_prover = RootProver::new::<Engine>(
        internal_recursive_vk.clone(),
        internal_recursive_pcs_data.clone(),
        root_system_params(),
        system_config.memory_config.memory_dimensions(),
        system_config.num_public_values,
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

    const AIR_MODIFIED_HEIGHT_IDX: usize = 2;
    trace_heights[AIR_MODIFIED_HEIGHT_IDX] *= 2;

    let root_prover = RootProver::new::<Engine>(
        internal_recursive_vk,
        internal_recursive_pcs_data,
        root_system_params(),
        system_config.memory_config.memory_dimensions(),
        system_config.num_public_values,
        Some(trace_heights.clone()),
    );
    let ctx = root_prover
        .generate_proving_ctx(internal_recursive_proof, &user_pvs_proof)
        .unwrap();

    for ((air_idx, air_ctx), expected_height) in ctx.per_trace.iter().zip(trace_heights) {
        assert_eq!(air_ctx.height(), expected_height, "air_idx {air_idx}");
    }
    let root_proof = root_prover.root_prove_from_ctx::<Engine>(ctx)?;

    let vk = root_prover.get_vk();
    let engine = Engine::new(vk.inner.params.clone());
    engine.verify(&vk, &root_proof)?;
    Ok(())
}
