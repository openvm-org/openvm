use std::{borrow::Borrow, path::PathBuf, sync::Arc};

use openvm_build::GuestOptions;
use openvm_circuit::{
    arch::{
        hasher::poseidon2::vm_poseidon2_hasher, ExecutionError, SingleSegmentVmExecutor,
        SystemConfig, VmConfig, VmExecutor,
    },
    system::{memory::tree::public_values::UserPublicValuesProof, program::trace::VmCommittedExe},
};
use openvm_native_circuit::{Native, NativeConfig};
use openvm_native_compiler::{conversion::CompilerOptions, prelude::*};
use openvm_native_recursion::{halo2::utils::CacheHalo2ParamsReader, types::InnerConfig};
use openvm_rv32im_transpiler::{Rv32ITranspilerExtension, Rv32MTranspilerExtension};
use openvm_sdk::{
    config::{AggConfig, AggStarkConfig, AppConfig, Halo2Config},
    keygen::AppProvingKey,
    verifier::{
        common::types::VmVerifierPvs,
        leaf::types::{LeafVmVerifierInput, UserPublicValuesRootProof},
    },
    Sdk, StdIn,
};
use openvm_stark_sdk::{
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        fri_params::standard_fri_params_with_100_bits_conjectured_security,
    },
    engine::{StarkEngine, StarkFriEngine},
    openvm_stark_backend::{p3_field::FieldAlgebra, Chip},
    p3_baby_bear::BabyBear,
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
        builder.compile_isa()
    };
    Sdk.commit_app_exe(
        standard_fri_params_with_100_bits_conjectured_security(app_log_blowup),
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
        leaf_fri_params: standard_fri_params_with_100_bits_conjectured_security(LEAF_LOG_BLOWUP),
        internal_fri_params: standard_fri_params_with_100_bits_conjectured_security(
            INTERNAL_LOG_BLOWUP,
        ),
        root_fri_params: standard_fri_params_with_100_bits_conjectured_security(ROOT_LOG_BLOWUP),
        profiling: false,
        compiler_options: CompilerOptions {
            enable_cycle_tracker: true,
            compile_prints: true,
            ..Default::default()
        },
    }
}

fn small_test_app_config(app_log_blowup: usize) -> AppConfig<NativeConfig> {
    AppConfig {
        app_fri_params: standard_fri_params_with_100_bits_conjectured_security(app_log_blowup)
            .into(),
        app_vm_config: NativeConfig::new(
            SystemConfig::default()
                .with_max_segment_len(200)
                .with_continuations()
                .with_public_values(16),
            Native,
        ),
        leaf_fri_params: standard_fri_params_with_100_bits_conjectured_security(LEAF_LOG_BLOWUP)
            .into(),
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
fn test_e2e_proof_generation_and_verification() {
    let app_log_blowup = 1;
    let app_config = small_test_app_config(app_log_blowup);
    let app_pk = Sdk.app_keygen(app_config).unwrap();
    let params_reader = CacheHalo2ParamsReader::new_with_default_params_dir();
    let agg_pk = Sdk
        .agg_keygen(agg_config_for_test(), &params_reader)
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
    assert!(Sdk.verify_evm_proof(&evm_verifier, &evm_proof));
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
