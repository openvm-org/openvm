use std::{borrow::Borrow, path::PathBuf, sync::Arc};

use openvm_build::GuestOptions;
use openvm_circuit::{
    arch::{
        hasher::poseidon2::vm_poseidon2_hasher, ContinuationVmProof, ExecutionError,
        SingleSegmentVmExecutor, SystemConfig, VmConfig, VmExecutor,
    },
    system::{memory::tree::public_values::UserPublicValuesProof, program::trace::VmCommittedExe},
};
use openvm_continuations::{
    static_verifier::StaticVerifierPvHandler,
    verifier::{
        common::types::{SpecialAirIds, VmVerifierPvs},
        leaf::types::{LeafVmVerifierInput, UserPublicValuesRootProof},
        root::types::RootVmVerifierPvs,
        utils::compress_babybear_var_to_bn254,
    },
};
use openvm_native_circuit::{Native, NativeConfig};
use openvm_native_compiler::{conversion::CompilerOptions, prelude::*};
use openvm_native_recursion::{
    config::outer::OuterConfig, halo2::utils::CacheHalo2ParamsReader, types::InnerConfig,
    vars::StarkProofVariable,
};
use openvm_rv32im_transpiler::{Rv32ITranspilerExtension, Rv32MTranspilerExtension};
use openvm_sdk::{
    codec::{Decode, Encode},
    commit::AppExecutionCommit,
    config::{AggConfig, AggStarkConfig, AppConfig, Halo2Config, SdkVmConfig},
    keygen::AppProvingKey,
    DefaultStaticVerifierPvHandler, Sdk, StdIn,
};
use openvm_stark_sdk::{
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        FriParameters,
    },
    engine::{StarkEngine, StarkFriEngine},
    openvm_stark_backend::{p3_field::FieldAlgebra, Chip},
    p3_baby_bear::BabyBear,
    p3_bn254_fr::Bn254Fr,
};
use openvm_transpiler::transpiler::Transpiler;

type SC = BabyBearPoseidon2Config;
type C = InnerConfig;
type F = BabyBear;

const NUM_PUB_VALUES: usize = 16;
const LEAF_LOG_BLOWUP: usize = 2;
const INTERNAL_LOG_BLOWUP: usize = 3;
const ROOT_LOG_BLOWUP: usize = 4;

fn run_leaf_verifier<VC: VmConfig<F>>(
    leaf_vm: &SingleSegmentVmExecutor<F, VC>,
    leaf_committed_exe: Arc<VmCommittedExe<SC>>,
    verifier_input: LeafVmVerifierInput<SC>,
) -> Result<Vec<F>, ExecutionError>
where
    VC::Executor: Chip<SC>,
    VC::Periphery: Chip<SC>,
{
    let exe_result = leaf_vm.execute_and_compute_heights(
        leaf_committed_exe.exe.clone(),
        verifier_input.write_to_stream(),
    )?;
    let runtime_pvs: Vec<_> = exe_result
        .public_values
        .iter()
        .map(|v| v.unwrap())
        .collect();
    Ok(runtime_pvs)
}

fn app_committed_exe_for_test(app_log_blowup: usize) -> Arc<VmCommittedExe<SC>> {
    let program = {
        let n = 200;
        let mut builder = Builder::<C>::default();
        let a: Felt<F> = builder.eval(F::ZERO);
        let b: Felt<F> = builder.eval(F::ONE);
        let c: Felt<F> = builder.uninit();
        builder.range(0, n).for_each(|_, builder| {
            builder.assign(&c, a + b);
            builder.assign(&a, b);
            builder.assign(&b, c);
        });
        builder.halt();
        let mut program = builder.compile_isa();
        program.max_num_public_values = NUM_PUB_VALUES;
        program
    };
    Sdk.commit_app_exe(
        FriParameters::new_for_testing(app_log_blowup),
        program.into(),
    )
    .unwrap()
}

fn agg_config_for_test() -> AggConfig {
    AggConfig {
        agg_stark_config: agg_stark_config_for_test(),
        halo2_config: Halo2Config {
            verifier_k: 24,
            wrapper_k: None,
            profiling: false,
        },
    }
}

fn agg_stark_config_for_test() -> AggStarkConfig {
    AggStarkConfig {
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

fn small_test_app_config(app_log_blowup: usize) -> AppConfig<NativeConfig> {
    AppConfig {
        app_fri_params: FriParameters::new_for_testing(app_log_blowup).into(),
        app_vm_config: NativeConfig::new(
            SystemConfig::default()
                .with_max_segment_len(200)
                .with_continuations()
                .with_public_values(NUM_PUB_VALUES),
            Native,
        ),
        leaf_fri_params: FriParameters::new_for_testing(LEAF_LOG_BLOWUP).into(),
        compiler_options: CompilerOptions {
            enable_cycle_tracker: true,
            ..Default::default()
        },
    }
}

#[test]
fn test_public_values_and_leaf_verification() {
    let app_log_blowup = 3;
    let app_config = small_test_app_config(app_log_blowup);
    let app_pk = AppProvingKey::keygen(app_config);
    let app_committed_exe = app_committed_exe_for_test(app_log_blowup);

    let agg_stark_config = agg_stark_config_for_test();
    let leaf_vm_config = agg_stark_config.leaf_vm_config();
    let leaf_vm = SingleSegmentVmExecutor::new(leaf_vm_config);
    let leaf_committed_exe = app_pk.leaf_committed_exe.clone();

    let app_engine = BabyBearPoseidon2Engine::new(app_pk.app_vm_pk.fri_params);
    let app_vm = VmExecutor::new(app_pk.app_vm_pk.vm_config.clone());
    let app_vm_result = app_vm
        .execute_and_generate_with_cached_program(app_committed_exe.clone(), vec![])
        .unwrap();
    assert!(app_vm_result.per_segment.len() > 2);

    let mut app_vm_seg_proofs: Vec<_> = app_vm_result
        .per_segment
        .into_iter()
        .map(|proof_input| app_engine.prove(&app_pk.app_vm_pk.vm_pk, proof_input))
        .collect();
    let app_last_proof = app_vm_seg_proofs.pop().unwrap();

    let expected_app_commit: [F; DIGEST_SIZE] = app_committed_exe.get_program_commit().into();

    // Verify all segments except the last one.
    let (first_seg_final_pc, first_seg_final_mem_root) = {
        let runtime_pvs = run_leaf_verifier(
            &leaf_vm,
            leaf_committed_exe.clone(),
            LeafVmVerifierInput {
                proofs: app_vm_seg_proofs.clone(),
                public_values_root_proof: None,
            },
        )
        .expect("failed to verify the first segment");

        let leaf_vm_pvs: &VmVerifierPvs<F> = runtime_pvs.as_slice().borrow();

        assert_eq!(leaf_vm_pvs.app_commit, expected_app_commit);
        assert_eq!(leaf_vm_pvs.connector.is_terminate, F::ZERO);
        assert_eq!(leaf_vm_pvs.connector.initial_pc, F::ZERO);
        (
            leaf_vm_pvs.connector.final_pc,
            leaf_vm_pvs.memory.final_root,
        )
    };

    let pv_proof = UserPublicValuesProof::compute(
        app_vm.config.system.memory_config.memory_dimensions(),
        NUM_PUB_VALUES,
        &vm_poseidon2_hasher(),
        app_vm_result.final_memory.as_ref().unwrap(),
    );
    let pv_root_proof = UserPublicValuesRootProof::extract(&pv_proof);

    // Verify the last segment with the correct public values root proof.
    {
        let runtime_pvs = run_leaf_verifier(
            &leaf_vm,
            leaf_committed_exe.clone(),
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
            &leaf_vm,
            leaf_committed_exe.clone(),
            LeafVmVerifierInput {
                proofs: vec![app_last_proof.clone()],
                public_values_root_proof: Some(wrong_pv_root_proof),
            },
        );
        assert!(
            matches!(execution_result, Err(ExecutionError::Fail { .. })),
            "Expected failure: the public value root proof has a wrong pv commit: {:?}",
            execution_result
        );
    }

    // Failure: The public value root proof has a wrong path proof.
    {
        let mut wrong_pv_root_proof = pv_root_proof.clone();
        wrong_pv_root_proof.sibling_hashes[0][0] += F::ONE;
        let execution_result = run_leaf_verifier(
            &leaf_vm,
            leaf_committed_exe.clone(),
            LeafVmVerifierInput {
                proofs: vec![app_last_proof.clone()],
                public_values_root_proof: Some(wrong_pv_root_proof),
            },
        );
        assert!(
            matches!(execution_result, Err(ExecutionError::Fail { .. })),
            "Expected failure: the public value root proof has a wrong path proof: {:?}",
            execution_result
        );
    }
}

#[test]
fn test_static_verifier_custom_pv_handler() {
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
    let app_pk = Sdk.app_keygen(app_config.clone()).unwrap();
    let app_committed_exe = app_committed_exe_for_test(app_log_blowup);
    println!("app_config: {:?}", app_config.app_vm_config);
    println!(
        "app_committed_exe max_num_public_values: {:?}",
        app_committed_exe.exe.program.max_num_public_values
    );
    let params_reader = CacheHalo2ParamsReader::new_with_default_params_dir();

    // Generate PK using custom PV handler
    println!("generate PK using custom PV handler");
    let commits = AppExecutionCommit::compute(
        &app_config.app_vm_config,
        &app_committed_exe,
        &app_pk.leaf_committed_exe,
    );
    let exe_commit = commits.exe_commit_to_bn254();
    let leaf_verifier_commit = commits.app_config_commit_to_bn254();

    let pv_handler = CustomPvHandler {
        exe_commit,
        leaf_verifier_commit,
    };
    let agg_pk = Sdk
        .agg_keygen(agg_config_for_test(), &params_reader, &pv_handler)
        .unwrap();

    // Generate verifier contract
    println!("generate verifier contract");
    let evm_verifier = Sdk
        .generate_snark_verifier_contract(&params_reader, &agg_pk)
        .unwrap();

    // Generate and verify proof
    println!("generate and verify proof");
    let evm_proof = Sdk
        .generate_evm_proof(
            &params_reader,
            Arc::new(app_pk),
            app_committed_exe,
            agg_pk,
            StdIn::default(),
        )
        .unwrap();
    assert!(Sdk.verify_evm_proof(&evm_verifier, &evm_proof).is_ok());
}

#[test]
fn test_e2e_proof_generation_and_verification() {
    let app_log_blowup = 1;
    let app_config = small_test_app_config(app_log_blowup);
    let app_pk = Sdk.app_keygen(app_config).unwrap();
    let params_reader = CacheHalo2ParamsReader::new_with_default_params_dir();
    let agg_pk = Sdk
        .agg_keygen(
            agg_config_for_test(),
            &params_reader,
            &DefaultStaticVerifierPvHandler,
        )
        .unwrap();
    let evm_verifier = Sdk
        .generate_snark_verifier_contract(&params_reader, &agg_pk)
        .unwrap();

    let evm_proof = Sdk
        .generate_evm_proof(
            &params_reader,
            Arc::new(app_pk),
            app_committed_exe_for_test(app_log_blowup),
            agg_pk,
            StdIn::default(),
        )
        .unwrap();
    assert!(Sdk.verify_evm_proof(&evm_verifier, &evm_proof).is_ok());
}

#[test]
fn test_sdk_guest_build_and_transpile() {
    let sdk = Sdk;
    let guest_opts = GuestOptions::default()
        // .with_features(vec!["zkvm"])
        // .with_options(vec!["--release"]);
        ;
    let mut pkg_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).to_path_buf();
    pkg_dir.push("guest");
    let one = sdk
        .build(guest_opts.clone(), &pkg_dir, &Default::default())
        .unwrap();
    let two = sdk
        .build(guest_opts.clone(), &pkg_dir, &Default::default())
        .unwrap();
    assert_eq!(one.instructions, two.instructions);
    assert_eq!(one.instructions, two.instructions);
    let transpiler = Transpiler::<F>::default()
        .with_extension(Rv32ITranspilerExtension)
        .with_extension(Rv32MTranspilerExtension);
    let _exe = sdk.transpile(one, transpiler).unwrap();
}

#[test]
fn test_inner_proof_codec_roundtrip() -> eyre::Result<()> {
    // generate a proof
    let sdk = Sdk;
    let mut pkg_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).to_path_buf();
    pkg_dir.push("guest");
    let elf = sdk.build(Default::default(), pkg_dir, &Default::default())?;
    let vm_config = SdkVmConfig::builder()
        .system(Default::default())
        .rv32i(Default::default())
        .rv32m(Default::default())
        .build();
    assert!(vm_config.system.config.continuation_enabled);
    let exe = sdk.transpile(elf, vm_config.transpiler())?;
    let fri_params = FriParameters::standard_fast();
    let app_config = AppConfig::new(fri_params, vm_config);
    let committed_exe = sdk.commit_app_exe(fri_params, exe)?;
    let app_pk = Arc::new(sdk.app_keygen(app_config)?);
    let app_proof = sdk.generate_app_proof(app_pk.clone(), committed_exe, StdIn::default())?;
    let mut app_proof_bytes = Vec::new();
    app_proof.encode(&mut app_proof_bytes)?;
    let decoded_app_proof = ContinuationVmProof::decode(&mut &app_proof_bytes[..])?;
    // Test decoding against derived serde implementation
    assert_eq!(
        serde_json::to_vec(&app_proof)?,
        serde_json::to_vec(&decoded_app_proof)?
    );
    // Test the decoding by verifying the decoded proof
    sdk.verify_app_proof(&app_pk.get_app_vk(), &decoded_app_proof)?;
    Ok(())
}
