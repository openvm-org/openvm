use std::{
    borrow::Borrow,
    path::PathBuf,
    sync::{Arc, OnceLock},
};

use eyre::Result;
use openvm_build::GuestOptions;
use openvm_circuit::{
    self,
    arch::{instructions::exe::VmExe, ContinuationVmProof, ExecutionError, VirtualMachineError},
    utils::test_system_config,
};
use openvm_continuations::verifier::{
    common::types::VmVerifierPvs,
    leaf::types::{LeafVmVerifierInput, UserPublicValuesRootProof},
};
use openvm_native_circuit::{execute_program_with_config, NativeConfig, NativeCpuBuilder};
use openvm_native_compiler::{conversion::CompilerOptions, prelude::*};
use openvm_sdk::{
    codec::{Decode, Encode},
    config::{AggregationConfig, AppConfig, SdkSystemConfig, SdkVmConfig},
    prover::verify_app_proof,
    Sdk, StdIn,
};
use openvm_stark_sdk::{
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        setup_tracing, FriParameters,
    },
    openvm_stark_backend::p3_field::FieldAlgebra,
    p3_baby_bear::BabyBear,
};
#[cfg(feature = "evm-verify")]
use {
    openvm_continuations::{
        static_verifier::StaticVerifierPvHandler,
        verifier::{
            common::types::SpecialAirIds, root::types::RootVmVerifierPvs,
            utils::compress_babybear_var_to_bn254,
        },
    },
    openvm_native_recursion::{
        config::outer::OuterConfig,
        halo2::{utils::Halo2ParamsReader, wrapper::Halo2WrapperProvingKey, RawEvmProof},
        vars::StarkProofVariable,
    },
    openvm_sdk::types::{EvmHalo2Verifier, EvmProof},
    openvm_stark_sdk::p3_bn254_fr::Bn254Fr,
    snark_verifier_sdk::evm::evm_verify,
};

type SC = BabyBearPoseidon2Config;
type F = BabyBear;

const NUM_PUB_VALUES: usize = 16;
const LEAF_LOG_BLOWUP: usize = 2;
const INTERNAL_LOG_BLOWUP: usize = 3;
const ROOT_LOG_BLOWUP: usize = 4;

/// `OpenVmHalo2Verifier` wraps the `snark-verifier` contract, meaning that
/// the default `fallback` interface can still be used. This function uses
/// the fallback interface as opposed to the `verify(..)` interface.
#[cfg(feature = "evm-verify")]
fn verify_evm_halo2_proof_with_fallback(
    openvm_verifier: &EvmHalo2Verifier,
    evm_proof: &EvmProof,
) -> Result<u64> {
    let evm_proof: RawEvmProof = evm_proof.clone().try_into()?;
    let gas_cost = evm_verify(
        openvm_verifier.artifact.bytecode.clone(),
        vec![evm_proof.instances.clone()],
        evm_proof.proof.clone(),
    )
    .map_err(|reason| eyre::eyre!("Sdk::verify_openvm_evm_proof: {reason:?}"))?;
    Ok(gas_cost)
}

fn run_leaf_verifier(
    leaf_vm_config: &NativeConfig,
    leaf_exe: &VmExe<F>,
    verifier_input: LeafVmVerifierInput<SC>,
) -> Result<Vec<F>, VirtualMachineError> {
    assert!(leaf_vm_config.system.has_public_values_chip());
    let (output, _vm) = execute_program_with_config::<BabyBearPoseidon2Engine, _>(
        leaf_exe.program.clone(),
        verifier_input.write_to_stream(),
        NativeCpuBuilder,
        leaf_vm_config.clone(),
    )?;
    Ok(output.system_records.public_values)
}

fn app_exe_for_test() -> Arc<VmExe<F>> {
    static EXE: OnceLock<Arc<VmExe<F>>> = OnceLock::new();
    EXE.get_or_init(|| {
        let sdk = Sdk::new(small_test_app_config(1)).unwrap();
        let mut pkg_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).to_path_buf();
        pkg_dir.push("guest/fib");
        let elf = sdk
            .build(Default::default(), pkg_dir, &Default::default(), None)
            .unwrap();
        sdk.convert_to_exe(elf).unwrap()
    })
    .clone()
}

fn agg_config_for_test() -> AggregationConfig {
    AggregationConfig {
        max_num_user_public_values: NUM_PUB_VALUES,
        leaf_fri_params: FriParameters::new_for_testing(LEAF_LOG_BLOWUP),
        internal_fri_params: FriParameters::new_for_testing(INTERNAL_LOG_BLOWUP),
        root_fri_params: FriParameters::new_for_testing(ROOT_LOG_BLOWUP),
        profiling: false,
        compiler_options: CompilerOptions {
            enable_cycle_tracker: true,
            ..Default::default()
        },
        root_max_constraint_degree: (1 << ROOT_LOG_BLOWUP) + 1,
    }
}

fn app_vm_config_for_test() -> SdkVmConfig {
    let config = test_system_config()
        .with_max_segment_len(200)
        .with_public_values(NUM_PUB_VALUES);
    SdkVmConfig::builder()
        .system(SdkSystemConfig { config })
        .rv32i(Default::default())
        .rv32m(Default::default())
        .io(Default::default())
        .build()
}

fn small_test_app_config(app_log_blowup: usize) -> AppConfig<SdkVmConfig> {
    AppConfig {
        app_fri_params: FriParameters::new_for_testing(app_log_blowup).into(),
        app_vm_config: app_vm_config_for_test(),
        leaf_fri_params: FriParameters::new_for_testing(LEAF_LOG_BLOWUP).into(),
        compiler_options: CompilerOptions {
            enable_cycle_tracker: true,
            ..Default::default()
        },
    }
}

#[test]
fn test_public_values_and_leaf_verification() -> eyre::Result<()> {
    setup_tracing();
    let app_log_blowup = 1;
    let app_config = small_test_app_config(app_log_blowup);
    let exe = app_exe_for_test();
    let pc_start = exe.pc_start;

    let agg_config = agg_config_for_test();
    let leaf_vm_config = agg_config.leaf_vm_config();

    let sdk = Sdk::new(app_config)?;
    let app_pk = sdk.app_pk();
    let leaf_exe = &app_pk.leaf_committed_exe.exe;
    let mut app_prover = sdk.app_prover(exe)?;
    let mut app_proof = app_prover.prove(StdIn::default())?;

    assert!(app_proof.per_segment.len() > 2);
    let app_last_proof = app_proof.per_segment.pop().unwrap();

    let expected_app_commit: [F; DIGEST_SIZE] = app_prover.app_program_commit().into();

    // Verify all segments except the last one.
    let (first_seg_final_pc, first_seg_final_mem_root) = {
        let runtime_pvs = run_leaf_verifier(
            &leaf_vm_config,
            leaf_exe,
            LeafVmVerifierInput {
                proofs: app_proof.per_segment.clone(),
                public_values_root_proof: None,
            },
        )
        .expect("failed to verify the first segment");

        let leaf_vm_pvs: &VmVerifierPvs<F> = runtime_pvs.as_slice().borrow();

        assert_eq!(leaf_vm_pvs.app_commit, expected_app_commit);
        assert_eq!(leaf_vm_pvs.connector.is_terminate, F::ZERO);
        assert_eq!(
            leaf_vm_pvs.connector.initial_pc,
            F::from_canonical_u32(pc_start)
        );
        (
            leaf_vm_pvs.connector.final_pc,
            leaf_vm_pvs.memory.final_root,
        )
    };

    let pv_proof = app_proof.user_public_values;
    let pv_root_proof = UserPublicValuesRootProof::extract(&pv_proof);

    // Verify the last segment with the correct public values root proof.
    {
        let runtime_pvs = run_leaf_verifier(
            &leaf_vm_config,
            leaf_exe,
            LeafVmVerifierInput {
                proofs: vec![app_last_proof.clone()],
                public_values_root_proof: Some(pv_root_proof.clone()),
            },
        )
        .expect("failed to verify the second segment");

        let leaf_vm_pvs: &VmVerifierPvs<F> = runtime_pvs.as_slice().borrow();
        assert_eq!(leaf_vm_pvs.app_commit, expected_app_commit);
        assert_eq!(leaf_vm_pvs.connector.initial_pc, first_seg_final_pc);
        assert_eq!(leaf_vm_pvs.connector.is_terminate, F::ONE);
        assert_eq!(leaf_vm_pvs.connector.exit_code, F::ZERO);
        assert_eq!(leaf_vm_pvs.memory.initial_root, first_seg_final_mem_root);
        assert_eq!(
            leaf_vm_pvs.public_values_commit,
            pv_root_proof.public_values_commit
        );
    }

    // Failure: The public value root proof has a wrong public values commit.
    {
        let mut wrong_pv_root_proof = pv_root_proof.clone();
        wrong_pv_root_proof.public_values_commit[0] += F::ONE;
        let execution_result = run_leaf_verifier(
            &leaf_vm_config,
            leaf_exe,
            LeafVmVerifierInput {
                proofs: vec![app_last_proof.clone()],
                public_values_root_proof: Some(wrong_pv_root_proof),
            },
        );
        assert!(
            matches!(
                execution_result,
                Err(VirtualMachineError::Execution(ExecutionError::Fail { .. }))
            ),
            "Expected failure: the public value root proof has a wrong pv commit: {:?}",
            execution_result
        );
    }

    // Failure: The public value root proof has a wrong path proof.
    {
        let mut wrong_pv_root_proof = pv_root_proof.clone();
        wrong_pv_root_proof.sibling_hashes[0][0] += F::ONE;
        let execution_result = run_leaf_verifier(
            &leaf_vm_config,
            leaf_exe,
            LeafVmVerifierInput {
                proofs: vec![app_last_proof.clone()],
                public_values_root_proof: Some(wrong_pv_root_proof),
            },
        );
        assert!(
            matches!(
                execution_result,
                Err(VirtualMachineError::Execution(ExecutionError::Fail { .. }))
            ),
            "Expected failure: the public value root proof has a wrong path proof: {:?}",
            execution_result
        );
    }
    Ok(())
}

#[cfg(feature = "evm-verify")]
#[test]
#[ignore = "slow"]
fn test_static_verifier_custom_pv_handler() -> eyre::Result<()> {
    use openvm_sdk::keygen::Halo2ProvingKey;

    // Define custom public values handler and implement StaticVerifierPvHandler trait on it
    pub struct CustomPvHandler {
        pub exe_commit: Bn254Fr,
        pub leaf_verifier_commit: Bn254Fr,
    }

    impl StaticVerifierPvHandler for CustomPvHandler {
        fn handle_public_values(
            &self,
            builder: &mut Builder<OuterConfig>,
            input: &StarkProofVariable<OuterConfig>,
            special_air_ids: &SpecialAirIds,
        ) -> usize {
            let pv_air = builder.get(&input.per_air, special_air_ids.public_values_air_id);
            let public_values: Vec<_> = pv_air
                .public_values
                .vec()
                .into_iter()
                .map(|x| builder.cast_felt_to_var(x))
                .collect();
            let pvs = RootVmVerifierPvs::from_flatten(public_values);
            let exe_commit = compress_babybear_var_to_bn254(builder, pvs.exe_commit);
            let leaf_commit = compress_babybear_var_to_bn254(builder, pvs.leaf_verifier_commit);
            let num_public_values = pvs.public_values.len();

            println!("num_public_values: {}", num_public_values);
            println!("self.exe_commit: {:?}", self.exe_commit);
            println!("self.leaf_verifier_commit: {:?}", self.leaf_verifier_commit);

            let expected_exe_commit: Var<Bn254Fr> = builder.constant(self.exe_commit);
            let expected_leaf_commit: Var<Bn254Fr> = builder.constant(self.leaf_verifier_commit);

            builder.assert_var_eq(exe_commit, expected_exe_commit);
            builder.assert_var_eq(leaf_commit, expected_leaf_commit);

            num_public_values
        }
    }

    // Test setup
    println!("test setup");
    let app_log_blowup = 1;
    let app_config = small_test_app_config(app_log_blowup);
    println!("app_config: {:?}", app_config.app_vm_config);
    let sdk = Sdk::new(app_config)?;
    let app_exe = app_exe_for_test();

    // Generate PK using custom PV handler
    println!("generate PK using custom PV handler");
    let app_commit = sdk.app_prover(app_exe.clone())?.app_commit();
    let exe_commit = app_commit.app_exe_commit.to_bn254();
    let leaf_verifier_commit = app_commit.app_vm_commit.to_bn254();

    let pv_handler = CustomPvHandler {
        exe_commit,
        leaf_verifier_commit,
    };
    let (agg_pk, dummy_internal_proof) = sdk.agg_pk_and_dummy_internal_proof();
    // SDK does not support CustomPvHandler, so we must use constructor directly
    let params_reader = sdk.halo2_params_reader();
    let halo2_pk = Halo2ProvingKey::keygen(
        *sdk.halo2_config(),
        params_reader,
        &pv_handler,
        agg_pk,
        dummy_internal_proof.clone(),
    )?;

    // Generate verifier contract
    println!("generate verifier contract");
    let wrapper_k = halo2_pk.wrapper.pinning.metadata.config_params.k;
    let params = params_reader.read_params(wrapper_k);
    let evm_verifier = halo2_pk.wrapper.generate_fallback_evm_verifier(&params);

    // Generate and verify proof
    println!("generate and verify proof");
    let _ = sdk.set_halo2_pk(halo2_pk).map_err(|_| panic!());
    let evm_proof = sdk.prove_evm(app_exe, StdIn::default())?;

    let evm_proof: RawEvmProof = evm_proof
        .clone()
        .try_into()
        .expect("failed to convert evm proof");
    Halo2WrapperProvingKey::evm_verify(&evm_verifier, &evm_proof).unwrap();
    Ok(())
}

#[cfg(feature = "evm-verify")]
#[test]
fn test_e2e_proof_generation_and_verification_with_pvs() -> eyre::Result<()> {
    let app_log_blowup = 1;
    let app_config = small_test_app_config(app_log_blowup);
    let mut sdk = Sdk::new(app_config)?;
    sdk.agg_config_mut().leaf_fri_params = FriParameters::new_for_testing(LEAF_LOG_BLOWUP);

    let evm_verifier = sdk.generate_halo2_verifier_solidity()?;
    let evm_proof = sdk.prove_evm(app_exe_for_test(), StdIn::default())?;

    verify_evm_halo2_proof_with_fallback(&evm_verifier, &evm_proof)?;
    Sdk::verify_evm_halo2_proof(&evm_verifier, evm_proof)?;
    Ok(())
}

#[test]
fn test_sdk_guest_build_and_transpile() -> eyre::Result<()> {
    let sdk = Sdk::new(small_test_app_config(1))?;
    let guest_opts = GuestOptions::default();
    let mut pkg_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).to_path_buf();
    pkg_dir.push("guest/fib");

    let one = sdk.build(guest_opts.clone(), &pkg_dir, &None, None)?;
    let two = sdk.build(guest_opts.clone(), &pkg_dir, &None, None)?;
    assert_eq!(one.instructions, two.instructions);
    assert_eq!(one.instructions, two.instructions);
    let _exe = sdk.convert_to_exe(one)?;
    Ok(())
}

#[test]
fn test_sdk_standard_with_p256() -> eyre::Result<()> {
    // WARNING: This test's keygen uses over the cargo test default stack
    // limit. To run this test, set env variable RUST_MIN_STACK=8388608.
    let sdk = Sdk::standard();
    let mut pkg_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).to_path_buf();
    pkg_dir.push("guest/p256");
    let elf = sdk.build(GuestOptions::default(), &pkg_dir, &None, None)?;
    let (proof, commit) = sdk.prove(elf, StdIn::default())?;
    Sdk::verify_proof(&sdk.agg_pk().get_agg_vk(), commit, &proof)?;
    Ok(())
}

#[test]
fn test_inner_proof_codec_roundtrip() -> eyre::Result<()> {
    // generate a proof
    let sdk = Sdk::new(small_test_app_config(1))?;
    assert!(sdk.app_config().app_vm_config.as_ref().continuation_enabled);
    let (_, app_vk) = sdk.app_keygen();
    let app_proof = sdk
        .app_prover(app_exe_for_test())?
        .prove(StdIn::default())?;
    let mut app_proof_bytes = Vec::new();
    app_proof.encode(&mut app_proof_bytes)?;
    let decoded_app_proof = ContinuationVmProof::decode(&mut &app_proof_bytes[..])?;
    // Test decoding against derived serde implementation
    assert_eq!(
        serde_json::to_vec(&app_proof)?,
        serde_json::to_vec(&decoded_app_proof)?
    );
    // Test the decoding by verifying the decoded proof
    verify_app_proof(&app_vk, &decoded_app_proof)?;
    Ok(())
}
