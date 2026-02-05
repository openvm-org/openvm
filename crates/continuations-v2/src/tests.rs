use std::sync::Arc;

use eyre::Result;
use itertools::Itertools;
use openvm_circuit::{
    arch::{
        ContinuationVmProver, VirtualMachine, VmCircuitConfig, VmInstance, instructions::exe::VmExe,
    },
    utils::test_utils::test_system_config,
};
use openvm_rv32im_circuit::{Rv32IConfig, Rv32ImBuilder, Rv32ImConfig};
use openvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use openvm_stark_backend::interaction::LogUpSecurityParameters;
use openvm_stark_sdk::{config::setup_tracing_with_log_level, p3_baby_bear::BabyBear};
use openvm_transpiler::{
    FromElf, elf::Elf, openvm_platform::memory::MEM_SIZE, transpiler::Transpiler,
};
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use stark_backend_v2::{
    F, StarkEngineV2, SystemParams, WhirConfig, WhirParams,
    keygen::types::MultiStarkVerifyingKeyV2, proof::Proof,
};
use tracing::Level;

use crate::aggregation::{AggregationProver, DEFAULT_MAX_NUM_PROOFS};

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use crate::aggregation::NonRootGpuProver as NonRootProver;
        use cuda_backend_v2::{BabyBearPoseidon2GpuEngineV2};
        type Engine = BabyBearPoseidon2GpuEngineV2;
    } else {
        use crate::aggregation::NonRootCpuProver as NonRootProver;
        use stark_backend_v2::{BabyBearPoseidon2CpuEngineV2, poseidon2::sponge::DuplexSponge};
        type Engine = BabyBearPoseidon2CpuEngineV2<DuplexSponge>;
    }
}

const LOG_MAX_TRACE_HEIGHT: usize = 20;

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

fn test_rv32im_config() -> Rv32ImConfig {
    Rv32ImConfig {
        rv32i: Rv32IConfig {
            system: test_system_config().with_max_segment_len(1 << LOG_MAX_TRACE_HEIGHT),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn run_leaf_aggregation(log_fib_input: usize) -> Result<(Arc<MultiStarkVerifyingKeyV2>, Proof)> {
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
    let leaf_proof = leaf_prover.agg_prove::<Engine>(
        &app_proof.per_segment,
        Some(app_proof.user_public_values.public_values_commit),
        false,
    )?;

    let leaf_vk = leaf_prover.get_vk();
    let engine = Engine::new(leaf_vk.inner.params.clone());
    engine.verify(&leaf_vk, &leaf_proof)?;
    Ok((leaf_vk, leaf_proof))
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
    let (leaf_vk, leaf_proof) = run_leaf_aggregation(10)?;

    let internal_for_leaf_prover = NonRootProver::<DEFAULT_MAX_NUM_PROOFS>::new::<Engine>(
        leaf_vk,
        internal_system_params(),
        false,
    );
    let internal_for_leaf_proof =
        internal_for_leaf_prover.agg_prove::<Engine>(&[leaf_proof], None, false)?;

    let internal_recursive_prover = NonRootProver::<DEFAULT_MAX_NUM_PROOFS>::new::<Engine>(
        internal_for_leaf_prover.get_vk(),
        internal_system_params(),
        true,
    );
    let mut internal_recursive_proof =
        internal_recursive_prover.agg_prove::<Engine>(&[internal_for_leaf_proof], None, false)?;

    let vk = internal_recursive_prover.get_vk();
    let engine = Engine::new(vk.inner.params.clone());

    for _internal_recursive_layer in 1..4 {
        internal_recursive_proof = internal_recursive_prover.agg_prove::<Engine>(
            &[internal_recursive_proof],
            None,
            true,
        )?;
        engine.verify(&vk, &internal_recursive_proof)?;
    }
    Ok(())
}
