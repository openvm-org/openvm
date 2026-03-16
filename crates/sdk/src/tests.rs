use std::{slice::from_ref, sync::Arc};

use eyre::Result;
use openvm::platform::memory::MEM_SIZE;
use openvm_circuit::arch::instructions::DEFERRAL_AS;
use openvm_deferral_circuit::DeferralFn;
use openvm_stark_backend::StarkEngine;
use openvm_stark_sdk::config::{
    app_params_with_100_bits_security, internal_params_with_100_bits_security,
    root_params_with_100_bits_security,
};
use openvm_transpiler::elf::Elf;
use openvm_verify_stark_circuit::extension::{
    get_deferral_state, get_raw_deferral_results, verify_stark_deferral_fn,
};
use openvm_verify_stark_host::vk::NonRootStarkVerifyingKey;

use crate::{
    config::{AggregationConfig, AggregationSystemParams, AppConfig, DEFAULT_APP_L_SKIP},
    prover::DeferralProver,
    CpuSdk, DeferralInput, Sdk, StdIn,
};

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use openvm_verify_stark_circuit::prover::DeferredVerifyGpuProver as VerifyProver;
        use openvm_verify_stark_circuit::prover::DeferredVerifyGpuCircuitProver as VerifyCircuitProver;
        type E = openvm_cuda_backend::BabyBearPoseidon2GpuEngine;
        type RootE = openvm_stark_sdk::config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2CpuEngine;
    } else {
        use openvm_verify_stark_circuit::prover::DeferredVerifyCpuProver as VerifyProver;
        use openvm_verify_stark_circuit::prover::DeferredVerifyCpuCircuitProver as VerifyCircuitProver;
        type E = openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2CpuEngine;
        type RootE = openvm_stark_sdk::config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2CpuEngine;
    }
}

#[test]
fn test_root_prover_trace_heights() -> Result<()> {
    let n_stack = 19;
    let app_params = app_params_with_100_bits_security(DEFAULT_APP_L_SKIP + n_stack);
    let agg_params = AggregationSystemParams::default();

    let elf = Elf::decode(
        include_bytes!("../programs/examples/fibonacci.elf"),
        MEM_SIZE as u32,
    )?;

    let sdk = Sdk::riscv32(app_params, agg_params);
    let app_exe = sdk.convert_to_exe(elf)?;
    let mut evm_prover = sdk.evm_prover(app_exe)?;

    let n = 1000u64;
    let mut stdin = StdIn::default();
    stdin.write(&n);

    let proof = evm_prover.prove(stdin)?;
    let vk = evm_prover.root_prover.0.get_vk();
    let engine = RootE::new(vk.inner.params.clone());
    engine.verify(&vk, &proof)?;

    Ok(())
}

#[test]
fn test_verify_stark_deferral() -> Result<()> {
    // ---- Step 1: Create a fibonacci proof ----
    let n_stack = 19;
    let app_params = app_params_with_100_bits_security(DEFAULT_APP_L_SKIP + n_stack);
    let agg_params = AggregationSystemParams::default();

    let fib_elf = Elf::decode(
        include_bytes!("../programs/examples/fibonacci.elf"),
        MEM_SIZE as u32,
    )?;

    let fib_sdk = Sdk::riscv32(app_params.clone(), agg_params.clone());
    let fib_exe = fib_sdk.convert_to_exe(fib_elf)?;

    let n = 100u64;
    let mut fib_stdin = StdIn::default();
    fib_stdin.write(&n);

    let (fib_proof, fib_baseline) = fib_sdk.prove(fib_exe, fib_stdin, &[])?;

    // ---- Step 2: Build the DeferredVerifyCircuitProver ----
    let fib_agg_prover = fib_sdk.agg_prover();
    let ir_vk = fib_agg_prover.internal_recursive_prover.get_vk();
    let ir_pcs_data = fib_agg_prover
        .internal_recursive_prover
        .get_self_vk_pcs_data()
        .unwrap();

    let fib_system_config = fib_sdk.app_config().app_vm_config.as_ref().clone();
    let memory_dimensions = fib_system_config.memory_config.memory_dimensions();
    let num_user_pvs = fib_system_config.num_public_values;

    let def_circuit_params = internal_params_with_100_bits_security();
    let deferred_verify_prover = VerifyProver::new::<E>(
        ir_vk,
        ir_pcs_data,
        def_circuit_params,
        memory_dimensions,
        num_user_pvs,
        None,
    );
    let verify_stark_prover = VerifyCircuitProver::new(deferred_verify_prover);

    // ---- Step 3: Create DeferralProver ----
    let hook_params = root_params_with_100_bits_security();
    let agg_config = AggregationConfig {
        params: agg_params.clone(),
    };
    let deferral_prover = DeferralProver::new(verify_stark_prover, agg_config, hook_params);

    // ---- Step 4: Create DeferralExtension ----
    let deferral_ext =
        deferral_prover.make_extension(vec![Arc::new(DeferralFn::new(verify_stark_deferral_fn))]);

    // ---- Step 5: Compute deferral state and guest stdin values ----
    let fib_vk = NonRootStarkVerifyingKey {
        mvk: fib_sdk.agg_vk().as_ref().clone(),
        baseline: fib_baseline,
    };

    // Get the raw results to extract input_commit and output for the guest stdin
    let raw_results = get_raw_deferral_results(&fib_vk, from_ref(&fib_proof))?;
    assert_eq!(raw_results.len(), 1);
    let input_commit: [u8; 32] = raw_results[0].input.clone().try_into().unwrap();
    let output_raw = &raw_results[0].output_raw;
    let app_exe_commit: [u8; 32] = output_raw[..32].try_into().unwrap();
    let app_vk_commit: [u8; 32] = output_raw[32..64].try_into().unwrap();
    let user_public_values = output_raw[64..].to_vec();

    // Build the deferral state for execution
    let deferral_state = get_deferral_state(&fib_vk, from_ref(&fib_proof), 0)?;

    // ---- Step 6: Create verify-stark SDK with deferral ----
    let mut vs_config = openvm_sdk_config::SdkVmConfig::riscv32();
    vs_config.deferral = Some(deferral_ext);
    vs_config.system.config.memory_config.addr_spaces[DEFERRAL_AS as usize].num_cells = 1 << 25;

    // TODO[INT-6241]: Switch this to SDK once CUDA is implemented for deferrals
    let vs_app_config = AppConfig::new(vs_config, app_params);
    let vs_sdk = CpuSdk::new(vs_app_config, agg_params)?.with_deferral_prover(deferral_prover);

    // ---- Step 7: Build the verify-stark ELF ----
    let programs_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("programs");
    let vs_elf = openvm_toolchain_tests::build_example_program_at_path(
        programs_dir,
        "verify-stark",
        &vs_sdk.app_config().app_vm_config,
    )?;
    let vs_exe = vs_sdk.convert_to_exe(vs_elf)?;

    // ---- Step 8: Set up stdin for the verify-stark guest program ----
    let mut vs_stdin = StdIn::default();
    vs_stdin.write(&app_exe_commit);
    vs_stdin.write(&app_vk_commit);
    vs_stdin.write(&user_public_values);
    vs_stdin.write(&input_commit);
    vs_stdin.deferrals = vec![deferral_state];

    // ---- Step 9: Create DeferralInput from the fibonacci proof ----
    let def_input = DeferralInput::from_inputs(&[fib_proof]);

    // ---- Step 10: Prove and verify ----
    let (vs_proof, vs_baseline) = vs_sdk.prove(vs_exe, vs_stdin, &[def_input])?;

    let vs_agg_vk = vs_sdk.agg_vk();
    Sdk::verify_proof(vs_agg_vk.as_ref().clone(), vs_baseline, &vs_proof)?;

    Ok(())
}
