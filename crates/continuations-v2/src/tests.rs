use std::sync::Arc;

use eyre::Result;
use itertools::Itertools;
use openvm_circuit::arch::ContinuationVmProver;
use openvm_circuit::{
    arch::{VirtualMachine, VmCircuitConfig, VmInstance, instructions::exe::VmExe},
    utils::test_utils::test_system_config,
};
use openvm_rv32im_circuit::{Rv32IConfig, Rv32ImBuilder, Rv32ImConfig};
use openvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use openvm_stark_sdk::config::setup_tracing_with_log_level;
use openvm_toolchain_tests::{build_example_program_at_path, get_programs_dir};
use openvm_transpiler::{FromElf, transpiler::Transpiler};
use p3_field::FieldAlgebra;
use stark_backend_v2::{F, StarkEngineV2, SystemParams, poseidon2::sponge::DuplexSponge};
use tracing::Level;

use crate::aggregation::AggregationProver;

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use crate::aggregation::NonRootGpuProver;
        use cuda_backend_v2::{BabyBearPoseidon2GpuEngineV2};
        type Engine = BabyBearPoseidon2GpuEngineV2<DuplexSponge>;
        type NonRootProver = NonRootGpuProver;
    } else {
        use crate::aggregation::NonRootCpuProver;
        use stark_backend_v2::BabyBearPoseidon2CpuEngineV2;
        type Engine = BabyBearPoseidon2CpuEngineV2<DuplexSponge>;
        type NonRootProver = NonRootCpuProver;
    }
}

const LOG_MAX_TRACE_HEIGHT: usize = 20;

const APP_SYSTEM_PARAMS: SystemParams = SystemParams {
    l_skip: 4,
    n_stack: 17,
    log_blowup: 1,
    k_whir: 4,
    num_whir_queries: 100,
    log_final_poly_len: 1,
    logup_pow_bits: 16,
    whir_pow_bits: 16,
};

const LEAF_SYSTEM_PARAMS: SystemParams = APP_SYSTEM_PARAMS;
const INTERNAL_SYSTEM_PARAMS: SystemParams = APP_SYSTEM_PARAMS;

fn test_rv32im_config() -> Rv32ImConfig {
    Rv32ImConfig {
        rv32i: Rv32IConfig {
            system: test_system_config().with_max_segment_len(1 << LOG_MAX_TRACE_HEIGHT),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn run_leaf_aggregation(log_fib_input: usize) -> Result<()> {
    let config = test_rv32im_config();
    let exe = VmExe::from_elf(
        build_example_program_at_path(get_programs_dir!(), "fibonacci", &config)?,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension),
    )?;
    let input = (1u64 << log_fib_input)
        .to_le_bytes()
        .map(F::from_canonical_u8)
        .to_vec();

    let engine = Engine::new(APP_SYSTEM_PARAMS);
    let (vm, app_pk) = VirtualMachine::new_with_keygen(engine, Rv32ImBuilder, config)?;
    let cached_program_trace = vm.commit_program_on_device(&exe.program);
    let mut instance = VmInstance::new(vm, exe.into(), cached_program_trace)?;
    let app_proof = instance.prove(vec![input])?;

    let leaf_prover =
        NonRootProver::new::<Engine>(Arc::new(app_pk.get_vk()), LEAF_SYSTEM_PARAMS, false);
    let leaf_proof = leaf_prover.agg_prove::<Engine>(
        &app_proof.per_segment,
        Some(app_proof.user_public_values.public_values_commit),
        false,
    )?;

    let leaf_vk = leaf_prover.get_vk();
    let engine = Engine::new(leaf_vk.inner.params);
    engine.verify(&leaf_vk, &leaf_proof)?;
    Ok(())
}

#[test]
fn test_single_segment_leaf_aggregation() -> Result<()> {
    setup_tracing_with_log_level(Level::INFO);
    run_leaf_aggregation(5)
}

#[test]
#[cfg(feature = "cuda")]
fn test_two_segments_leaf_aggregation() -> Result<()> {
    // This test is too slow to run on CPU
    setup_tracing_with_log_level(Level::INFO);
    run_leaf_aggregation(17)
}

#[test]
fn test_internal_recursive_vk_stabilization() -> Result<()> {
    setup_tracing_with_log_level(Level::INFO);
    let config = test_rv32im_config();

    let engine = Engine::new(APP_SYSTEM_PARAMS);
    let (_, app_vk) = engine.keygen(&config.create_airs()?.into_airs().collect_vec());

    let leaf_prover = NonRootProver::new::<Engine>(Arc::new(app_vk), LEAF_SYSTEM_PARAMS, false);
    let internal_0_prover =
        NonRootProver::new::<Engine>(leaf_prover.get_vk(), INTERNAL_SYSTEM_PARAMS, false);
    let internal_1_prover =
        NonRootProver::new::<Engine>(internal_0_prover.get_vk(), INTERNAL_SYSTEM_PARAMS, false);

    // The internal vk should stabilize at the second internal layer
    let test_prover =
        NonRootProver::new::<Engine>(internal_1_prover.get_vk(), INTERNAL_SYSTEM_PARAMS, true);
    assert_eq!(
        test_prover.get_commit(),
        test_prover.self_vk_pcs_data.unwrap().commitment
    );
    Ok(())
}
