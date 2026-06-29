use std::{path::Path, slice::from_ref};

use eyre::Result;
use openvm::platform::memory::MEM_SIZE;
#[cfg(feature = "root-prover")]
use openvm_continuations::prover::DeferralCircuitProver;
use openvm_sdk_config::{
    deferral::{DeferralConfig, SupportedDeferral},
    SdkVmConfig,
};
#[cfg(feature = "root-prover")]
use openvm_stark_backend::{codec::Encode, StarkEngine};
use openvm_stark_backend::{SystemParams, WhirProximityStrategy};
use openvm_stark_sdk::{
    config::{
        app_params_with_100_bits_security, hook_params_with_100_bits_security,
        internal_params_with_100_bits_security, leaf_params_with_100_bits_security,
        params_with_100_bits_security, root_params_with_100_bits_security,
        RECURSION_MAX_CONSTRAINT_DEGREE,
    },
    utils::setup_tracing,
};
use openvm_transpiler::elf::Elf;
use openvm_verify_stark_circuit::extension::{get_deferral_state, get_raw_deferral_results};
use openvm_verify_stark_host::{
    vk::{VerificationBaseline, VmStarkVerifyingKey},
    VmStarkProof,
};
use serde::Serialize;

#[cfg(feature = "root-prover")]
use crate::prover::DeferralProof;
use crate::{
    builder::GenericSdkBuilder,
    config::{AggregationConfig, AggregationSystemParams, AppConfig, DEFAULT_APP_L_SKIP},
    prover::{DeferralAggProver, MultiDeferralCircuitProver},
    DeferralInput, Sdk, StdIn,
};

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use openvm_verify_stark_circuit::prover::DeferredVerifyGpuProver as VerifyProver;
        use openvm_verify_stark_circuit::prover::DeferredVerifyGpuCircuitProver as VerifyCircuitProver;
        type E = openvm_cuda_backend::BabyBearPoseidon2GpuEngine;
        #[cfg(feature = "root-prover")]
        type RootE = openvm_cuda_backend::BabyBearBn254Poseidon2GpuEngine;
    } else {
        use openvm_verify_stark_circuit::prover::DeferredVerifyCpuProver as VerifyProver;
        use openvm_verify_stark_circuit::prover::DeferredVerifyCpuCircuitProver as VerifyCircuitProver;
        type E = openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2CpuEngine;
        #[cfg(feature = "root-prover")]
        type RootE = openvm_stark_sdk::config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2CpuEngine;
    }
}

fn get_params() -> (SystemParams, AggregationSystemParams, SystemParams) {
    let n_stack = 19;
    let app_params = get_params_from_env("APP_PARAMS_OVERRIDE", || {
        app_params_with_100_bits_security(DEFAULT_APP_L_SKIP + n_stack)
    });
    let agg_params = AggregationSystemParams {
        leaf: get_params_from_env("LEAF_PARAMS_OVERRIDE", || {
            leaf_params_with_100_bits_security()
        }),
        internal: get_params_from_env("INTERNAL_PARAMS_OVERRIDE", || {
            internal_params_with_100_bits_security()
        }),
    };
    let root_params = get_params_from_env("ROOT_PARAMS_OVERRIDE", || {
        root_params_with_100_bits_security()
    });

    (app_params, agg_params, root_params)
}

/// Creates a fibonacci SDK with standard test parameters.
fn make_fib_sdk() -> (Sdk, SystemParams, AggregationSystemParams) {
    let (app_params, agg_params, _root_params) = get_params(); // get_overriden_params(test_json);
    let mut sdk_builder =
        GenericSdkBuilder::new().app_config(AppConfig::riscv32(app_params.clone()));
    sdk_builder = sdk_builder.agg_params(agg_params.clone());
    #[cfg(feature = "root-prover")]
    {
        sdk_builder = sdk_builder.root_params(_root_params);
    }
    (sdk_builder.build().unwrap(), app_params, agg_params)
}

fn get_params_from_env(env_var: &str, default: impl FnOnce() -> SystemParams) -> SystemParams {
    match std::env::var(env_var) {
        Ok(s) => {
            eprintln!("getting params from env {env_var}");
            serde_json::from_str(&s).unwrap()
        }
        Err(_) => default(),
    }
}

/// Generates a fibonacci VM STARK proof using the given SDK.
fn generate_fib_vm_stark_proof(fib_sdk: &Sdk) -> Result<(VmStarkProof, VerificationBaseline)> {
    let fib_elf = Elf::decode(
        include_bytes!("../programs/examples/fibonacci.elf"),
        MEM_SIZE as u32,
    )?;
    let fib_exe = fib_sdk.convert_to_exe(fib_elf)?;
    let n = 100u64;
    let mut stdin = StdIn::default();
    stdin.write(&n);
    Ok(fib_sdk.prove(fib_exe, stdin, &[])?)
}

fn riscv32_config_with_deferral(deferral: DeferralConfig) -> SdkVmConfig {
    SdkVmConfig::builder()
        .system(Default::default())
        .rv32i(Default::default())
        .rv32m(Default::default())
        .io(Default::default())
        .deferral(deferral)
        .build()
        .optimize()
}

fn make_verify_stark_circuit_prover(
    sdk: &Sdk,
    def_circuit_params: SystemParams,
    def_idx: usize,
) -> VerifyCircuitProver {
    let agg_prover = sdk.agg_prover();
    let ir_vk = agg_prover.internal_recursive_prover.get_vk();
    let ir_pcs_data = agg_prover
        .internal_recursive_prover
        .get_self_vk_pcs_data()
        .unwrap();
    let system_config = sdk.app_config().app_vm_config.as_ref().clone();
    let memory_dimensions = system_config.memory_config.memory_dimensions();
    let num_user_pvs = system_config.num_public_values;
    let deferred_verify_prover = VerifyProver::new::<E>(
        ir_vk,
        ir_pcs_data.commitment.into(),
        def_circuit_params,
        memory_dimensions,
        num_user_pvs,
        None,
        def_idx,
    );
    VerifyCircuitProver::new(deferred_verify_prover)
}

/// Builds a MultiDeferralCircuitProver from a base SDK with `num_deferral_circuits` copies of the
/// verify-stark deferral circuit.
fn make_multi_deferral_circuit_prover_with_count(
    sdk: &Sdk,
    agg_params: &AggregationSystemParams,
    num_deferral_circuits: usize,
) -> MultiDeferralCircuitProver {
    assert!(num_deferral_circuits > 0);
    let def_circuit_params = agg_params.internal.clone();
    let verify_stark_prover = make_verify_stark_circuit_prover(sdk, def_circuit_params.clone(), 0);
    let hook_params = hook_params_with_100_bits_security();
    let agg_config = AggregationConfig {
        params: agg_params.clone(),
    };
    let mut multi_deferral_circuit_prover =
        MultiDeferralCircuitProver::new(verify_stark_prover, agg_config, hook_params);
    for def_idx in 1..num_deferral_circuits {
        multi_deferral_circuit_prover = multi_deferral_circuit_prover.with_prover(
            make_verify_stark_circuit_prover(sdk, def_circuit_params.clone(), def_idx),
        );
    }
    multi_deferral_circuit_prover
}

/// Builds a deferral-enabled riscv32 SDK whose App VM inventory includes the
/// deferral periphery chips (DeferralPoseidon2Chip, count chip, etc.).
fn make_deferral_enabled_sdk(
    fib_sdk: &Sdk,
    app_params: SystemParams,
    agg_params: AggregationSystemParams,
) -> Result<Sdk> {
    make_deferral_enabled_sdk_with_count(fib_sdk, app_params, agg_params, 1)
}

fn make_deferral_enabled_sdk_with_count(
    fib_sdk: &Sdk,
    app_params: SystemParams,
    agg_params: AggregationSystemParams,
    num_deferral_circuits: usize,
) -> Result<Sdk> {
    let multi_deferral_circuit_prover =
        make_multi_deferral_circuit_prover_with_count(fib_sdk, &agg_params, num_deferral_circuits);
    let supported_deferrals = vec![SupportedDeferral::VerifyStark; num_deferral_circuits];
    let deferral_config = multi_deferral_circuit_prover.make_config(supported_deferrals);

    let vm_config = riscv32_config_with_deferral(deferral_config);

    Ok(Sdk::builder()
        .app_config(AppConfig::new(vm_config, app_params))
        .agg_params(agg_params)
        .multi_deferral_circuit_prover(multi_deferral_circuit_prover)
        .build()?)
}

fn make_verify_stark_path_sdk(
    app_params: SystemParams,
    agg_params: AggregationSystemParams,
) -> Result<Sdk> {
    let vm_config = SdkVmConfig::riscv32();
    let memory_dimensions = vm_config.system.config.memory_config.memory_dimensions();
    let num_user_pvs = vm_config.system.config.num_public_values;
    let deferral_agg_prover = DeferralAggProver::verify_stark(
        &agg_params,
        hook_params_with_100_bits_security(),
        memory_dimensions,
        num_user_pvs,
    );
    let deferral_config = deferral_agg_prover
        .multi_deferral_circuit_prover
        .make_config(vec![SupportedDeferral::VerifyStark]);
    let vm_config = riscv32_config_with_deferral(deferral_config);

    Ok(Sdk::builder()
        .app_config(AppConfig::new(vm_config, app_params))
        .agg_params(agg_params)
        .deferral_agg_prover(deferral_agg_prover)
        .build()?)
}

fn make_verify_stark_inputs(
    child_sdk: &Sdk,
    child_proof: &VmStarkProof,
    child_baseline: VerificationBaseline,
) -> Result<(StdIn, DeferralInput)> {
    let (stdin, mut def_inputs) =
        make_verify_stark_inputs_for_indices(child_sdk, child_proof, child_baseline, &[0], 1)?;
    Ok((stdin, def_inputs.pop().unwrap()))
}

fn make_verify_stark_inputs_for_indices(
    child_sdk: &Sdk,
    child_proof: &VmStarkProof,
    child_baseline: VerificationBaseline,
    present_def_indices: &[usize],
    num_deferral_circuits: usize,
) -> Result<(StdIn, Vec<DeferralInput>)> {
    let child_vk = VmStarkVerifyingKey {
        mvk: child_sdk.agg_vk().as_ref().clone(),
        baseline: child_baseline,
    };

    let raw_results = get_raw_deferral_results(&child_vk, from_ref(child_proof))?;
    assert_eq!(raw_results.len(), 1);
    let input_commit: [u8; 32] = raw_results[0].input.clone().try_into().unwrap();
    let output_raw = &raw_results[0].output_raw;
    let app_exe_commit: [u8; 32] = output_raw[..32].try_into().unwrap();
    let app_vm_commit: [u8; 32] = output_raw[32..64].try_into().unwrap();

    let user_public_values = collapse_user_public_values(&output_raw[64..]);

    let mut stdin = StdIn::default();
    stdin.write(&app_exe_commit);
    stdin.write(&app_vm_commit);
    stdin.write(&user_public_values);
    stdin.write(&input_commit);
    stdin.deferrals = vec![Default::default(); num_deferral_circuits];

    let proof_input = DeferralInput::from_inputs(from_ref(child_proof));
    let mut def_inputs = vec![DeferralInput::default(); num_deferral_circuits];
    for &def_idx in present_def_indices {
        assert!(def_idx < num_deferral_circuits);
        stdin.deferrals[def_idx] =
            get_deferral_state(&child_vk, from_ref(child_proof), def_idx as u32)?;
        def_inputs[def_idx] = proof_input.clone();
    }

    Ok((stdin, def_inputs))
}

fn collapse_user_public_values(expanded: &[u8]) -> Vec<u8> {
    const F_NUM_BYTES: usize = 4;
    assert!(expanded.len().is_multiple_of(F_NUM_BYTES));
    expanded
        .chunks_exact(F_NUM_BYTES)
        .map(|bytes| {
            assert_eq!(&bytes[1..], &[0; F_NUM_BYTES - 1]);
            bytes[0]
        })
        .collect()
}

/// Builds a deferral-enabled verify-stark SDK from a fibonacci SDK and proof.
///
/// Returns the SDK, the verify-stark stdin, and the deferral input.
fn make_deferral_sdk(
    fib_sdk: &Sdk,
    fib_proof: VmStarkProof,
    fib_baseline: VerificationBaseline,
    app_params: SystemParams,
    agg_params: AggregationSystemParams,
) -> Result<(Sdk, StdIn, DeferralInput)> {
    let (vs_stdin, def_input) = make_verify_stark_inputs(fib_sdk, &fib_proof, fib_baseline)?;
    let vs_sdk = make_deferral_enabled_sdk(fib_sdk, app_params, agg_params)?;

    Ok((vs_sdk, vs_stdin, def_input))
}

#[derive(Serialize, Default)]
struct ParamSet {
    app: Option<SystemParams>,
    leaf: Option<SystemParams>,
    root: Option<SystemParams>,
    internal: Option<SystemParams>,
}

#[test]
fn generate_interesting_internal_params() {
    let w_stack = 512;
    let make_internal_params =
        |max_log_height, k_whir, log_blowup, l_skip, pow_bits, folding_pow_bits| -> SystemParams {
            let n_stack = max_log_height - l_skip;
            let proximity = WhirProximityStrategy::ListDecoding { m: 2 };
            params_with_100_bits_security(
                log_blowup,
                l_skip,
                n_stack,
                w_stack,
                folding_pow_bits,
                pow_bits,
                proximity,
                RECURSION_MAX_CONSTRAINT_DEGREE,
                pow_bits,
                k_whir,
            )
        };
    // Root override matching internal_sweep_root.json: fixed l_skip=2, n_stack=18, log_blowup=4,
    // k_whir=4, proximity=ListDecoding{m:1}, all pow_bits=20; only w_stack varies per entry.
    let make_root_params = |root_w_stack| -> SystemParams {
        let max_log_height = 20;
        let l_skip = 2;
        let n_stack = max_log_height - l_skip;
        let log_blowup = 4;
        let k_whir = 4;
        let proximity = WhirProximityStrategy::ListDecoding { m: 1 };
        let pow_bits = 20;
        params_with_100_bits_security(
            log_blowup,
            l_skip,
            n_stack,
            root_w_stack,
            pow_bits,
            pow_bits,
            proximity,
            RECURSION_MAX_CONSTRAINT_DEGREE,
            pow_bits,
            k_whir,
        )
    };
    let make_param =
        |max_log_height, k_whir, log_blowup, l_skip, pow_bits, folding_pow_bits| ParamSet {
            internal: Some(make_internal_params(
                max_log_height,
                k_whir,
                log_blowup,
                l_skip,
                pow_bits,
                folding_pow_bits,
            )),
            ..Default::default()
        };
    let make_param_with_root =
        |max_log_height, k_whir, log_blowup, l_skip, pow_bits, folding_pow_bits, root_w_stack| {
            ParamSet {
                internal: Some(make_internal_params(
                    max_log_height,
                    k_whir,
                    log_blowup,
                    l_skip,
                    pow_bits,
                    folding_pow_bits,
                )),
                root: Some(make_root_params(root_w_stack)),
                ..Default::default()
            }
        };
    let no_root_params = vec![
        // log_blowup=2, k_whir=3, max_log_height=19
        make_param(19, 3, 2, 1, 20, 18),
        make_param(19, 3, 2, 2, 20, 18),
        make_param(19, 3, 2, 3, 20, 18),
        make_param(19, 3, 2, 4, 20, 18),
        make_param(19, 3, 2, 5, 20, 18),
        // log_blowup=3, k_whir=3, max_log_height=19
        make_param(19, 3, 3, 1, 20, 18),
        make_param(19, 3, 3, 2, 20, 18),
        make_param(19, 3, 3, 3, 20, 18),
        make_param(19, 3, 3, 4, 20, 18),
        make_param(19, 3, 3, 5, 20, 18),
        // log_blowup=1, k_whir=4, max_log_height=19
        make_param(19, 4, 1, 1, 20, 18),
        make_param(19, 4, 1, 2, 20, 18),
        make_param(19, 4, 1, 3, 20, 18),
        make_param(19, 4, 1, 4, 20, 18),
        make_param(19, 4, 1, 5, 20, 18),
        // log_blowup=2, k_whir=4, max_log_height=19
        make_param(19, 4, 2, 1, 20, 18),
        make_param(19, 4, 2, 2, 20, 18),
        make_param(19, 4, 2, 3, 20, 18),
        make_param(19, 4, 2, 4, 20, 18),
        make_param(19, 4, 2, 5, 20, 18),
        // log_blowup=3, k_whir=4, max_log_height=19
        make_param(19, 4, 3, 1, 20, 18),
        make_param(19, 4, 3, 2, 20, 18),
        make_param(19, 4, 3, 3, 20, 18),
        make_param(19, 4, 3, 4, 20, 18),
        make_param(19, 4, 3, 5, 20, 18),
        // log_blowup=1, k_whir=4, max_log_height=20
        make_param(20, 4, 1, 1, 20, 18),
        make_param(20, 4, 1, 2, 20, 18),
        make_param(20, 4, 1, 3, 20, 18),
        make_param(20, 4, 1, 4, 20, 18),
        make_param(20, 4, 1, 5, 20, 18),
        // log_blowup=2, k_whir=4, max_log_height=20
        make_param(20, 4, 2, 1, 20, 18),
        make_param(20, 4, 2, 2, 20, 18),
        make_param(20, 4, 2, 3, 20, 18),
        make_param(20, 4, 2, 4, 20, 18),
        make_param(20, 4, 2, 5, 20, 18),
    ];
    let root_params = vec![
        // internal log_blowup=3, k_whir=4, max_log_height=19; root w_stack=18
        make_param_with_root(19, 4, 3, 1, 20, 18, 18),
        make_param_with_root(19, 4, 3, 2, 20, 18, 18),
        make_param_with_root(19, 4, 3, 3, 20, 18, 18),
        make_param_with_root(19, 4, 3, 4, 20, 18, 18),
        make_param_with_root(19, 4, 3, 5, 20, 18, 18),
        // internal log_blowup=2, k_whir=4, max_log_height=20; root w_stack=33
        make_param_with_root(20, 4, 2, 1, 20, 18, 33),
        make_param_with_root(20, 4, 2, 2, 20, 18, 33),
        make_param_with_root(20, 4, 2, 3, 20, 18, 33),
        make_param_with_root(20, 4, 2, 4, 20, 18, 33),
        make_param_with_root(20, 4, 2, 5, 20, 18, 33),
    ];

    let tests_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let no_root_path = tests_dir.join("internal_sweep_no_root.json");
    serde_json::to_writer_pretty(
        std::fs::File::create(&no_root_path).expect("failed to create internal_sweep_no_root.json"),
        &no_root_params,
    )
    .expect("failed to write internal_sweep_no_root.json");
    println!(
        "wrote {} entries to {}",
        no_root_params.len(),
        no_root_path.display()
    );

    let root_path = tests_dir.join("internal_sweep_root.json");
    serde_json::to_writer_pretty(
        std::fs::File::create(&root_path).expect("failed to create internal_sweep_root.json"),
        &root_params,
    )
    .expect("failed to write internal_sweep_root.json");
    println!(
        "wrote {} entries to {}",
        root_params.len(),
        root_path.display()
    );
}

#[test]
fn generate_interesting_root_params() {
    let max_log_height = 20;
    let w_stack = 18;
    let make_param = |k_whir, log_blowup, l_skip, pow_bits| {
        let n_stack = max_log_height - l_skip;
        let proximity = WhirProximityStrategy::ListDecoding { m: 1 };
        let root = params_with_100_bits_security(
            log_blowup,
            l_skip,
            n_stack,
            w_stack,
            pow_bits,
            pow_bits,
            proximity,
            RECURSION_MAX_CONSTRAINT_DEGREE,
            pow_bits,
            k_whir,
        );
        ParamSet {
            root: Some(root),
            ..Default::default()
        }
    };
    let good_params = vec![
        // k_whir = 4
        make_param(4, 2, 1, 20),
        make_param(4, 2, 4, 20),
        make_param(4, 2, 5, 20),
        make_param(4, 3, 1, 20),
        make_param(4, 3, 2, 20),
        make_param(4, 3, 3, 20),
        make_param(4, 3, 5, 20),
        // k_whir = 3
        make_param(3, 4, 1, 20),
        make_param(3, 4, 2, 20),
        make_param(3, 4, 4, 20),
        make_param(3, 3, 5, 20),
        make_param(3, 3, 1, 20),
        make_param(3, 3, 3, 20),
        make_param(3, 3, 5, 20),
        // k_whir = 4, pow_bits lowered
        make_param(4, 4, 2, 15),
        make_param(4, 4, 2, 18),
    ];

    let output_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("root_params.json");
    let file = std::fs::File::create(&output_path).expect("failed to create root_params.json");
    serde_json::to_writer_pretty(file, &good_params).expect("failed to write root_params.json");
    println!(
        "wrote {} good params to {}",
        good_params.len(),
        output_path.display()
    );
}

#[test]
fn test_sdk_fibonacci() -> Result<()> {
    setup_tracing();
    let (sdk, _, _) = make_fib_sdk();
    let elf = Elf::decode(
        include_bytes!("../programs/examples/fibonacci.elf"),
        MEM_SIZE as u32,
    )?;
    let app_exe = sdk.convert_to_exe(elf.clone())?;

    let n = 1000u64;
    let mut stdin = StdIn::default();
    stdin.write(&n);

    #[cfg(not(feature = "evm-verify"))]
    {
        #[cfg(feature = "root-prover")]
        {
            let mut evm_prover = sdk.evm_prover_without_halo2(app_exe)?;
            let proof = evm_prover.prove_root(stdin, &[])?;
            let vk = evm_prover.root_prover.0.get_vk();
            let engine = RootE::new(vk.inner.params.clone());
            engine.verify(&vk, &proof)?;
        }
        #[cfg(not(feature = "root-prover"))]
        {
            let (proof, baseline) = sdk.prove(app_exe, stdin, &[])?;
            Sdk::verify_proof((*sdk.agg_vk()).clone(), baseline, &proof)?;
        }
    }
    #[cfg(feature = "evm-verify")]
    {
        let app_commit = sdk.app_commit(app_exe.clone())?;
        let evm_proof = sdk.prove_evm(app_exe, stdin, &[])?;
        let openvm_verifier = sdk.generate_halo2_verifier_solidity()?;
        let _gas_cost = Sdk::verify_evm_halo2_proof(&openvm_verifier, evm_proof, Some(app_commit))?;
    }
    Ok(())
}

#[test]
fn test_verify_stark_deferral() -> Result<()> {
    setup_tracing();
    let (fib_sdk, app_params, agg_params) = make_fib_sdk();
    let (fib_proof, fib_baseline) = generate_fib_vm_stark_proof(&fib_sdk)?;
    let (vs_sdk, vs_stdin, def_input) =
        make_deferral_sdk(&fib_sdk, fib_proof, fib_baseline, app_params, agg_params)?;

    let vs_elf = Elf::decode(
        include_bytes!("../programs/examples/verify-stark.elf"),
        MEM_SIZE as u32,
    )?;
    let vs_exe = vs_sdk.convert_to_exe(vs_elf)?;

    #[cfg(feature = "evm-verify")]
    {
        let app_commit = vs_sdk.app_commit(vs_exe.clone())?;
        let mut evm_prover = vs_sdk.evm_prover(vs_exe)?;
        let vs_proof = evm_prover.prove_evm(vs_stdin, &[def_input])?;

        let openvm_verifier = vs_sdk.generate_halo2_verifier_solidity()?;
        let _gas_cost = Sdk::verify_evm_halo2_proof(&openvm_verifier, vs_proof, Some(app_commit))?;
    }
    #[cfg(not(feature = "evm-verify"))]
    {
        #[cfg(feature = "root-prover")]
        {
            let mut evm_prover = vs_sdk.evm_prover_without_halo2(vs_exe)?;
            let vs_proof = evm_prover.prove_root(vs_stdin, &[def_input])?;

            let vk = evm_prover.root_prover.0.get_vk();
            let engine = RootE::new(vk.inner.params.clone());
            engine.verify(&vk, &vs_proof)?;
        }
        #[cfg(not(feature = "root-prover"))]
        {
            let (proof, baseline) = vs_sdk.prove(vs_exe, vs_stdin, &[def_input])?;
            Sdk::verify_proof((*vs_sdk.agg_vk()).clone(), baseline, &proof)?;
        }
    }

    Ok(())
}

#[test]
fn test_verify_many_deferrals() -> Result<()> {
    setup_tracing();
    const NUM_DEFERRAL_CIRCUITS: usize = 5;

    let (fib_sdk, app_params, agg_params) = make_fib_sdk();
    let (fib_proof, fib_baseline) = generate_fib_vm_stark_proof(&fib_sdk)?;
    let (vs_stdin, def_inputs) = make_verify_stark_inputs_for_indices(
        &fib_sdk,
        &fib_proof,
        fib_baseline,
        &[0, 1, 3, 4],
        NUM_DEFERRAL_CIRCUITS,
    )?;
    let vs_sdk = make_deferral_enabled_sdk_with_count(
        &fib_sdk,
        app_params,
        agg_params,
        NUM_DEFERRAL_CIRCUITS,
    )?;

    let vs_elf = Elf::decode(
        include_bytes!("../programs/examples/verify-many.elf"),
        MEM_SIZE as u32,
    )?;
    let vs_exe = vs_sdk.convert_to_exe(vs_elf)?;

    #[cfg(feature = "evm-verify")]
    {
        let app_commit = vs_sdk.app_commit(vs_exe.clone())?;
        let evm_proof = vs_sdk.prove_evm(vs_exe, vs_stdin, &def_inputs)?;
        let openvm_verifier = vs_sdk.generate_halo2_verifier_solidity()?;
        let _gas_cost = Sdk::verify_evm_halo2_proof(&openvm_verifier, evm_proof, Some(app_commit))?;
    }

    #[cfg(not(feature = "evm-verify"))]
    {
        let (vs_proof, vs_baseline) = vs_sdk.prove(vs_exe, vs_stdin, &def_inputs)?;
        assert!(
            vs_proof.deferral_merkle_proofs.is_some(),
            "verify-many proof must carry deferral merkle proofs",
        );
        Sdk::verify_proof(vs_sdk.agg_vk().as_ref().clone(), vs_baseline, &vs_proof)?;
    }

    Ok(())
}

#[cfg(feature = "root-prover")]
#[test]
fn test_verify_stark_with_deferral_child() -> Result<()> {
    setup_tracing();
    let (fib_sdk, app_params, agg_params) = make_fib_sdk();
    let (fib_proof, fib_baseline) = generate_fib_vm_stark_proof(&fib_sdk)?;
    let (vs_sdk, vs_stdin, def_input) = make_deferral_sdk(
        &fib_sdk,
        fib_proof,
        fib_baseline,
        app_params,
        agg_params.clone(),
    )?;

    let vs_elf = Elf::decode(
        include_bytes!("../programs/examples/verify-stark.elf"),
        MEM_SIZE as u32,
    )?;
    let vs_exe = vs_sdk.convert_to_exe(vs_elf)?;

    let (vs_proof, _) = vs_sdk.prove(vs_exe, vs_stdin, &[def_input])?;
    assert!(
        vs_proof.deferral_merkle_proofs.is_some(),
        "deferral-enabled verify-stark child proof must carry deferral merkle proofs",
    );
    let expected_def_hook_commit = vs_sdk
        .def_hook_commit()
        .expect("deferral-enabled SDK should expose a deferral hook commit");

    // ---- Step 5: Feed the encoded proof through the trait adapter ----
    let vs_agg_prover = vs_sdk.agg_prover();
    let vs_ir_vk = vs_agg_prover.internal_recursive_prover.get_vk();
    let vs_ir_pcs_data = vs_agg_prover
        .internal_recursive_prover
        .get_self_vk_pcs_data()
        .unwrap();
    let vs_system_config = vs_sdk.app_config().app_vm_config.as_ref().clone();

    // This nested verifier is intentionally constructed in deferral-aware mode because the
    // verify-stark child proof above was itself produced through a deferral-enabled SDK.
    let nested_verify_prover = VerifyProver::new::<E>(
        vs_ir_vk,
        vs_ir_pcs_data.commitment.into(),
        agg_params.internal.clone(),
        vs_system_config.memory_config.memory_dimensions(),
        vs_system_config.num_public_values,
        Some(expected_def_hook_commit.into()),
        0,
    );
    let nested_verify_circuit_prover = VerifyCircuitProver::new(nested_verify_prover);

    let encoded_vs_proof = vs_proof.encode_to_vec()?;
    let nested_def_proof = nested_verify_circuit_prover.prove(&encoded_vs_proof);

    let vk = nested_verify_circuit_prover.get_vk();
    let engine = E::new(vk.inner.params.clone());
    engine.verify(&vk, &nested_def_proof)?;

    Ok(())
}

#[test]
fn test_verify_stark_path_sdk_can_verify_own_proofs() -> Result<()> {
    setup_tracing();
    let (app_params, agg_params, _) = get_params();
    let sdk = make_verify_stark_path_sdk(app_params, agg_params)?;
    let agg_vk = sdk.agg_vk().as_ref().clone();

    let vs_elf = Elf::decode(
        include_bytes!("../programs/examples/verify-stark.elf"),
        MEM_SIZE as u32,
    )?;
    let vs_exe = sdk.convert_to_exe(vs_elf)?;

    let (fib_proof, fib_baseline) = generate_fib_vm_stark_proof(&sdk)?;
    assert!(fib_proof.deferral_merkle_proofs.is_some(),);
    Sdk::verify_proof(agg_vk.clone(), fib_baseline.clone(), &fib_proof)?;

    let (vs_stdin, def_input) = make_verify_stark_inputs(&sdk, &fib_proof, fib_baseline)?;
    let (vs_proof, vs_baseline) = sdk.prove(vs_exe.clone(), vs_stdin, &[def_input])?;
    assert!(vs_proof.deferral_merkle_proofs.is_some(),);
    Sdk::verify_proof(agg_vk.clone(), vs_baseline.clone(), &vs_proof)?;

    let (vs2_stdin, vs2_def_input) = make_verify_stark_inputs(&sdk, &vs_proof, vs_baseline)?;
    #[cfg(feature = "evm-verify")]
    {
        let app_commit = sdk.app_commit(vs_exe.clone())?;
        let evm_proof = sdk.prove_evm(vs_exe, vs2_stdin, &[vs2_def_input])?;

        let openvm_verifier = sdk.generate_halo2_verifier_solidity()?;
        let _gas_cost = Sdk::verify_evm_halo2_proof(&openvm_verifier, evm_proof, Some(app_commit))?;
    }

    #[cfg(not(feature = "evm-verify"))]
    {
        let (vs2_proof, vs2_baseline) = sdk.prove(vs_exe, vs2_stdin, &[vs2_def_input])?;
        assert!(vs2_proof.deferral_merkle_proofs.is_some(),);
        Sdk::verify_proof(agg_vk, vs2_baseline, &vs2_proof)?;
    }

    Ok(())
}

#[cfg(feature = "root-prover")]
#[test]
fn test_deferrals_enabled_without_usage() -> Result<()> {
    setup_tracing();
    let (fib_sdk, app_params, agg_params) = make_fib_sdk();
    let sdk = make_deferral_enabled_sdk(&fib_sdk, app_params, agg_params)?;

    let elf = Elf::decode(
        include_bytes!("../programs/examples/fibonacci.elf"),
        MEM_SIZE as u32,
    )?;
    let app_exe = sdk.convert_to_exe(elf)?;

    let n = 1000u64;
    let mut stdin = StdIn::default();
    stdin.write(&n);

    let mut evm_prover = sdk.evm_prover_without_halo2(app_exe)?;
    let proof = evm_prover.prove_root(stdin, &[])?;

    // ---- Step 3: Verify the final result ----
    let vk = evm_prover.root_prover.0.get_vk();
    let engine = RootE::new(vk.inner.params.clone());
    engine.verify(&vk, &proof)?;

    Ok(())
}

#[cfg(feature = "root-prover")]
#[test]
fn test_prove_mixed_vm_def_depth_mismatch() -> Result<()> {
    setup_tracing();
    let (fib_sdk, app_params, agg_params) = make_fib_sdk();
    let (fib_proof, fib_baseline) = generate_fib_vm_stark_proof(&fib_sdk)?;
    let (vs_sdk, vs_stdin, def_input) =
        make_deferral_sdk(&fib_sdk, fib_proof, fib_baseline, app_params, agg_params)?;

    let vs_elf = Elf::decode(
        include_bytes!("../programs/examples/verify-stark.elf"),
        MEM_SIZE as u32,
    )?;
    let vs_exe = vs_sdk.convert_to_exe(vs_elf)?;

    // ---- Step 1: Generate base VM and deferral proofs ----
    let agg_prover = vs_sdk.agg_prover();
    let app_proof = vs_sdk.app_prover(vs_exe)?.prove(vs_stdin)?;
    let (vm_proof, mut internal_layer_metadata) = agg_prover.prove_vm(app_proof)?;

    // We assume that the verify-stark program is small enough where only a single
    // internal_recursive layer is needed to fully aggregate its proof.
    assert_eq!(internal_layer_metadata.internal_recursive_layer, 1);

    let def_prover = vs_sdk.def_agg_prover.unwrap();
    let def_hook_proofs = def_prover
        .multi_deferral_circuit_prover
        .prove(&[def_input])?;
    let (def_proof, mut def_internal_recursive_layer) =
        def_prover.agg_prover.prove_def(def_hook_proofs)?;
    assert_eq!(def_internal_recursive_layer, 1);

    // ---- Step 2: Generate mixed proof with wrapped VM proof ----
    let mut wrapped_vm_metadata = internal_layer_metadata.clone();
    let mut wrapped_vm_proof = vm_proof.clone();
    for _ in 0..2 {
        wrapped_vm_proof = agg_prover.wrap_proof(wrapped_vm_proof, &mut wrapped_vm_metadata)?;
    }
    let wrapped_vm_mixed_proof = agg_prover.prove_mixed(
        wrapped_vm_proof,
        def_proof.clone(),
        &mut wrapped_vm_metadata,
        def_internal_recursive_layer,
    )?;

    // ---- Step 3: Generate mixed proof with wrapped deferral proof ----
    let wrapped_def_proof = match def_proof {
        DeferralProof::Present(mut p) => {
            for _ in 0..2 {
                p = agg_prover.wrap_def_inner(p, def_internal_recursive_layer)?;
                def_internal_recursive_layer += 1;
            }
            DeferralProof::Present(p)
        }
        DeferralProof::Absent(_) => panic!("expected DeferralProof::Present"),
    };
    let wrapped_def_mixed_proof = agg_prover.prove_mixed(
        vm_proof,
        wrapped_def_proof,
        &mut internal_layer_metadata,
        def_internal_recursive_layer,
    )?;

    // ---- Step 4: Verify mixed proofs ----
    let vk = agg_prover.internal_recursive_prover.get_vk();
    let engine = E::new(vk.inner.params.clone());
    engine.verify(&vk, &wrapped_vm_mixed_proof.inner)?;
    engine.verify(&vk, &wrapped_def_mixed_proof.inner)?;

    Ok(())
}

/// Cell-count profiling test for the static verifier circuit using a production root proof.
///
/// Root verifier params match `pipeline_cell_count_profiling` in static-verifier crate.
/// The root proof is generated from a full SDK aggregation pipeline and cached to disk.
///
/// Run with:
/// ```sh
/// OPENVM_CACHE_DIR=cache OPENVM_PROFILE_DIR=profile \
///   cargo nextest run --cargo-profile=fast -p openvm-sdk --features cuda,cell-profiling \
///   -- sdk_static_verifier_cell_profiling
/// ```
#[cfg(feature = "cell-profiling")]
#[cfg(feature = "root-prover")]
#[test]
fn sdk_static_verifier_cell_profiling() -> Result<()> {
    use std::path::Path;

    use halo2_base::gates::circuit::{builder::BaseCircuitBuilder, CircuitBuilderStage};
    use openvm::platform::memory::MEM_SIZE;
    use openvm_continuations::{CommitBytes, RootSC};
    use openvm_stark_backend::{
        codec::{Decode, Encode},
        proof::Proof,
    };
    use openvm_static_verifier::{
        compute_dag_onion_commit,
        field::baby_bear::{BabyBearChip, BabyBearExtChip},
        log_heights_per_air_from_proof, StaticVerifierCircuit,
    };

    use crate::{
        config::{AggregationSystemParams, DEFAULT_APP_L_SKIP},
        keygen::dummy::compute_root_proof_heights,
        prover::{EvmProver, RootProver},
        Sdk, StdIn,
    };

    // Root verifier params matching pipeline_cell_count_profiling in static-verifier
    let (app_params, agg_params, root_params) = get_params();
    let cache_dir = std::env::var("OPENVM_CACHE_DIR").unwrap_or_else(|_| "cache".to_string());
    std::fs::create_dir_all(&cache_dir)?;

    let proof_path = format!("{cache_dir}/sdk_root_proof.bin");
    let vk_path = format!("{cache_dir}/sdk_root_vk.bin");
    let commit_path = format!("{cache_dir}/sdk_onion_commit.bin");

    let (root_vk, root_proof, onion_commit) =
        if Path::new(&proof_path).exists() && Path::new(&vk_path).exists() {
            eprintln!("Loading cached root proof from {cache_dir}/");
            let proof_bytes = std::fs::read(&proof_path)?;
            let root_proof = Proof::<RootSC>::decode_from_bytes(&proof_bytes)?;

            let vk_bytes = std::fs::read(&vk_path)?;
            let root_vk = bitcode::deserialize(&vk_bytes)
                .map_err(|e| eyre::eyre!("failed to deserialize root VK: {e}"))?;

            let commit_bytes: [u8; 32] = std::fs::read(&commit_path)?
                .try_into()
                .map_err(|_| eyre::eyre!("invalid commit file"))?;
            let onion_commit = CommitBytes::new(commit_bytes).into();

            (root_vk, root_proof, onion_commit)
        } else {
            eprintln!("Generating root proof via SDK pipeline (this takes a while)...");
            let n_stack = 19;

            let elf = Elf::decode(
                include_bytes!("../programs/examples/fibonacci.elf"),
                MEM_SIZE as u32,
            )?;
            let sdk = Sdk::riscv32(app_params, agg_params);
            let app_exe = sdk.convert_to_exe(elf)?;

            // Compute trace heights for root prover with profiling params
            let system_config = sdk.app_config().app_vm_config.as_ref();
            let agg_prover = sdk.agg_prover();
            let (trace_heights, root_pk) = compute_root_proof_heights::<E, _>(
                sdk.app_vm_builder().clone(),
                &sdk.app_pk().app_vm_pk,
                agg_prover.clone(),
                root_params.clone(),
                None,
            )?;

            let ir_vk = agg_prover.internal_recursive_prover.get_vk();
            let ir_pcs_data = agg_prover
                .internal_recursive_prover
                .get_self_vk_pcs_data()
                .unwrap();
            let vk_commit: CommitBytes = ir_pcs_data.commitment.into();
            let onion_commit = compute_dag_onion_commit(&ir_vk);

            let memory_dimensions = system_config.memory_config.memory_dimensions();
            let num_user_pvs = system_config.num_public_values;

            let root_prover = std::sync::Arc::new(RootProver::from_pk(
                ir_vk,
                vk_commit,
                root_pk,
                memory_dimensions,
                num_user_pvs,
                None,
                Some(trace_heights),
            ));

            let mut evm_prover = EvmProver::<E, _>::new(
                sdk.app_vm_builder().clone(),
                &sdk.app_pk().app_vm_pk,
                app_exe,
                agg_prover,
                None,
                root_prover.clone(),
                None,
            )?;

            let n = 100u64;
            let mut stdin = StdIn::default();
            stdin.write(&n);

            let root_proof = evm_prover.prove_root(stdin, &[])?;
            let root_vk_arc = root_prover.0.get_vk();
            let root_vk = root_vk_arc.as_ref().clone();

            // Verify the root proof
            let engine = RootE::new(root_vk.inner.params.clone());
            engine.verify(&root_vk, &root_proof)?;

            // Cache to disk
            eprintln!("Caching root proof to {cache_dir}/");
            std::fs::write(&proof_path, root_proof.encode_to_vec()?)?;
            std::fs::write(
                &vk_path,
                bitcode::serialize(&root_vk)
                    .map_err(|e| eyre::eyre!("failed to serialize root VK: {e}"))?,
            )?;
            std::fs::write(&commit_path, CommitBytes::from(onion_commit).as_slice())?;

            (root_vk, root_proof, onion_commit)
        };

    // Run static verifier cell profiling
    eprintln!("Running static verifier cell profiling...");
    let log_heights = log_heights_per_air_from_proof(&root_proof);

    let circuit = StaticVerifierCircuit::try_new(root_vk, onion_commit, &log_heights)
        .expect("Failed to construct StaticVerifierCircuit");

    let profile_dir = std::env::var("OPENVM_PROFILE_DIR").unwrap_or_else(|_| "profile".to_string());
    std::env::set_var("OPENVM_PROFILE_DIR", &profile_dir);

    let mut builder = BaseCircuitBuilder::from_stage(CircuitBuilderStage::Mock)
        .use_k(22)
        .use_lookup_bits(21)
        .use_instance_columns(0);
    let range = builder.range_chip();
    let ext_chip = BabyBearExtChip::new(BabyBearChip::new(std::sync::Arc::new(range)));
    let ctx = builder.main(0);

    let initial_cells = ctx.advice.len();
    circuit.populate_verify_stark_constraints(ctx, &ext_chip, &root_proof);
    let final_cells = ctx.advice.len();
    eprintln!(
        "Static verifier cell count: {} (delta: {})",
        final_cells,
        final_cells - initial_cells
    );
    assert!(
        final_cells > initial_cells,
        "expected advice cells to increase"
    );

    Ok(())
}
