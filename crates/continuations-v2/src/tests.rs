use std::sync::Arc;

use eyre::Result;
use openvm_circuit::arch::ContinuationVmProver;
use openvm_circuit::{
    arch::{VirtualMachine, VmInstance, instructions::exe::VmExe},
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
use stark_backend_v2::{
    BabyBearPoseidon2CpuEngineV2, F, StarkEngineV2, SystemParams, poseidon2::sponge::DuplexSponge,
};
use test_case::test_case;
use tracing::Level;

use crate::aggregation::NonRootVerifier;

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

const LEAF_SYSTEM_PARAMS: SystemParams = SystemParams {
    l_skip: 4,
    n_stack: 17,
    log_blowup: 1,
    k_whir: 4,
    num_whir_queries: 100,
    log_final_poly_len: 1,
    logup_pow_bits: 16,
    whir_pow_bits: 16,
};

fn test_rv32im_config() -> Rv32ImConfig {
    Rv32ImConfig {
        rv32i: Rv32IConfig {
            system: test_system_config().with_max_segment_len(1 << LOG_MAX_TRACE_HEIGHT),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test_case(5 ; "single segment")]
// #[test_case(17 ; "several segments")]
fn test_leaf_aggregation(log_fib_height: usize) -> Result<()> {
    setup_tracing_with_log_level(Level::INFO);
    let config = test_rv32im_config();
    let exe = VmExe::from_elf(
        build_example_program_at_path(get_programs_dir!(), "fibonacci", &config)?,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension),
    )?;
    let input = (1u64 << log_fib_height)
        .to_le_bytes()
        .map(F::from_canonical_u8)
        .to_vec();

    let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(APP_SYSTEM_PARAMS);
    let (vm, app_pk) = VirtualMachine::new_with_keygen(engine, Rv32ImBuilder, config)?;
    let cached_program_trace = vm.commit_program_on_device(&exe.program);
    let mut instance = VmInstance::new(vm, exe.into(), cached_program_trace)?;
    let app_proof = instance.prove(vec![input])?;

    let leaf_verifier = NonRootVerifier::<2>::new(Arc::new(app_pk.get_vk()), LEAF_SYSTEM_PARAMS);
    let leaf_proof = leaf_verifier.verify(
        &app_proof.per_segment,
        Some(app_proof.user_public_values.public_values_commit),
    )?;

    let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(leaf_verifier.vk.inner.params);
    engine.verify(&leaf_verifier.vk, &leaf_proof)?;
    Ok(())
}
