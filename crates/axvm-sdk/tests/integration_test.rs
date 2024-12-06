use std::borrow::Borrow;

use ax_stark_sdk::{
    ax_stark_backend::{p3_field::AbstractField, prover::types::Proof, Chip},
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        fri_params::standard_fri_params_with_100_bits_conjectured_security,
    },
    engine::{StarkEngine, StarkFriEngine},
};
use axvm_circuit::{
    arch::{
        hasher::poseidon2::vm_poseidon2_hasher, ExecutionError, SingleSegmentVmExecutor,
        SystemConfig, VmConfig, VmExecutor,
    },
    system::memory::tree::public_values::UserPublicValuesProof,
};
use axvm_native_circuit::{Native, NativeConfig};
use axvm_native_compiler::{conversion::CompilerOptions, prelude::*};
use axvm_native_recursion::types::InnerConfig;
use axvm_sdk::{
    commit::AppExecutionCommit,
    config::{AggConfig, AppConfig, SdkVmConfig},
    keygen::{AggProvingKey, AppProvingKey},
    prover::{commit_app_exe, generate_leaf_committed_exe, StarkProver},
    verifier::{
        common::types::VmVerifierPvs,
        leaf::types::{LeafVmVerifierInput, UserPublicValuesRootProof},
        root::types::RootVmVerifierPvs,
    },
    Sdk, StdIn,
};
use p3_baby_bear::BabyBear;
use utils::{assert_agg_config_eq, assert_agg_pk_eq};

mod utils;

type SC = BabyBearPoseidon2Config;
type C = InnerConfig;
type F = BabyBear;

const NUM_PUB_VALUES: usize = 16;

// TODO: keygen agg_pk once for all IT tests and store in a file
fn load_agg_pk_into_e2e_prover<VC: VmConfig<F>>(
    app_config: AppConfig<VC>,
) -> (StarkProver<VC>, Proof<SC>)
where
    VC::Executor: Chip<SC>,
    VC::Periphery: Chip<SC>,
{
    let agg_config = AggConfig {
        max_num_user_public_values: NUM_PUB_VALUES,
        leaf_fri_params: standard_fri_params_with_100_bits_conjectured_security(4),
        internal_fri_params: standard_fri_params_with_100_bits_conjectured_security(3),
        root_fri_params: standard_fri_params_with_100_bits_conjectured_security(2),
        compiler_options: CompilerOptions {
            enable_cycle_tracker: true,
            compile_prints: true,
            ..Default::default()
        },
    };

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

    let app_committed_exe = commit_app_exe(&app_config, program);
    let app_pk = AppProvingKey::keygen(app_config);
    let leaf_committed_exe = generate_leaf_committed_exe(&agg_config, &app_pk);
    let (agg_pk, dummy) = AggProvingKey::dummy_proof_and_keygen(agg_config);
    (
        StarkProver::new(app_pk, app_committed_exe)
            .with_agg_pk_and_leaf_committed_exe(agg_pk, leaf_committed_exe),
        dummy,
    )
}

fn run_leaf_verifier<VC: VmConfig<F>>(
    verifier_input: LeafVmVerifierInput<SC>,
    e2e_prover: &StarkProver<VC>,
) -> Result<Vec<F>, ExecutionError>
where
    VC::Executor: Chip<SC>,
    VC::Periphery: Chip<SC>,
{
    let leaf_vm = SingleSegmentVmExecutor::new(e2e_prover.agg_pk().leaf_vm_pk.vm_config.clone());
    let exe_result = leaf_vm.execute(
        e2e_prover.leaf_committed_exe().exe.clone(),
        verifier_input.write_to_stream(),
    )?;
    let runtime_pvs: Vec<_> = exe_result
        .public_values
        .iter()
        .map(|v| v.unwrap())
        .collect();
    Ok(runtime_pvs)
}

fn small_test_app_config(log_blowup_factor: usize) -> AppConfig<NativeConfig> {
    AppConfig {
        app_fri_params: standard_fri_params_with_100_bits_conjectured_security(log_blowup_factor),
        app_vm_config: NativeConfig::new(
            SystemConfig::default()
                .with_max_segment_len(200)
                .with_continuations()
                .with_public_values(16),
            Native,
        ),
    }
}

#[test]
fn test_public_values_and_leaf_verification() {
    let app_config = small_test_app_config(3);
    let (e2e_prover, _) = load_agg_pk_into_e2e_prover(app_config);

    let app_engine = BabyBearPoseidon2Engine::new(e2e_prover.app_pk.app_vm_pk.fri_params);
    let app_vm = VmExecutor::new(e2e_prover.app_pk.app_vm_pk.vm_config.clone());
    let app_vm_result = app_vm
        .execute_and_generate_with_cached_program(e2e_prover.app_committed_exe.clone(), vec![])
        .unwrap();
    assert!(app_vm_result.per_segment.len() > 2);

    let mut app_vm_seg_proofs: Vec<_> = app_vm_result
        .per_segment
        .into_iter()
        .map(|proof_input| app_engine.prove(&e2e_prover.app_pk.app_vm_pk.vm_pk, proof_input))
        .collect();
    let app_last_proof = app_vm_seg_proofs.pop().unwrap();

    let expected_app_commit: [F; DIGEST_SIZE] =
        e2e_prover.app_committed_exe.get_program_commit().into();

    // Verify all segments except the last one.
    let (first_seg_final_pc, first_seg_final_mem_root) = {
        let runtime_pvs = run_leaf_verifier(
            LeafVmVerifierInput {
                proofs: app_vm_seg_proofs.clone(),
                public_values_root_proof: None,
            },
            &e2e_prover,
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
        e2e_prover.agg_pk().num_public_values(),
        &vm_poseidon2_hasher(),
        app_vm_result.final_memory.as_ref().unwrap(),
    );
    let pv_root_proof = UserPublicValuesRootProof::extract(&pv_proof);

    // Verify the last segment with the correct public values root proof.
    {
        let runtime_pvs = run_leaf_verifier(
            LeafVmVerifierInput {
                proofs: vec![app_last_proof.clone()],
                public_values_root_proof: Some(pv_root_proof.clone()),
            },
            &e2e_prover,
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
            LeafVmVerifierInput {
                proofs: vec![app_last_proof.clone()],
                public_values_root_proof: Some(wrong_pv_root_proof),
            },
            &e2e_prover,
        );
        match execution_result.err().unwrap() {
            ExecutionError::Fail { .. } => {}
            _ => {
                panic!("Expected failure: the public value root proof has a wrong pv commit")
            }
        }
    }

    // Failure: The public value root proof has a wrong path proof.
    {
        let mut wrong_pv_root_proof = pv_root_proof.clone();
        wrong_pv_root_proof.sibling_hashes[0][0] += F::ONE;
        let execution_result = run_leaf_verifier(
            LeafVmVerifierInput {
                proofs: vec![app_last_proof.clone()],
                public_values_root_proof: Some(wrong_pv_root_proof),
            },
            &e2e_prover,
        );
        match execution_result.err().unwrap() {
            ExecutionError::Fail { .. } => {}
            _ => panic!("Expected failure: the public value root proof has a wrong path proof"),
        }
    }
}

#[test]
fn test_e2e_proof_generation() {
    let app_config = small_test_app_config(3);
    let (e2e_prover, dummy_internal_proof) = load_agg_pk_into_e2e_prover(app_config);

    let air_id_perm = e2e_prover.agg_pk().root_verifier_pk.air_id_permutation();
    let special_air_ids = air_id_perm.get_special_air_ids();

    let root_proof = e2e_prover.generate_e2e_proof(StdIn::default());
    let root_pvs = RootVmVerifierPvs::from_flatten(
        root_proof.per_air[special_air_ids.public_values_air_id]
            .public_values
            .clone(),
    );

    let app_exe_commit = AppExecutionCommit::compute(
        &e2e_prover.app_pk.app_vm_pk.vm_config,
        &e2e_prover.app_committed_exe,
        e2e_prover.leaf_committed_exe(),
    );

    assert_eq!(root_pvs.exe_commit, app_exe_commit.exe_commit);
    assert_eq!(
        root_pvs.leaf_verifier_commit,
        app_exe_commit.leaf_vm_verifier_commit
    );

    static_verifier::test_static_verifier(
        &e2e_prover.agg_pk().root_verifier_pk,
        dummy_internal_proof,
        &root_proof,
    );
}

#[test]
fn test_e2e_app_log_blowup_1() {
    let app_config = small_test_app_config(1);

    let (e2e_prover, dummy_internal_proof) = load_agg_pk_into_e2e_prover(app_config);
    let root_proof = e2e_prover.generate_e2e_proof(StdIn::default());

    static_verifier::test_static_verifier(
        &e2e_prover.agg_pk().root_verifier_pk,
        dummy_internal_proof,
        &root_proof,
    );
}

#[test]
fn test_agg_keygen_store_and_load() {
    const AGG_PK_PATH: &str = "temp/agg_pk.out";

    let sdk = Sdk;
    let agg_config = AggConfig {
        max_num_user_public_values: NUM_PUB_VALUES,
        leaf_fri_params: standard_fri_params_with_100_bits_conjectured_security(4),
        internal_fri_params: standard_fri_params_with_100_bits_conjectured_security(3),
        root_fri_params: standard_fri_params_with_100_bits_conjectured_security(2),
        compiler_options: CompilerOptions {
            enable_cycle_tracker: true,
            compile_prints: true,
            ..Default::default()
        },
    };

    let (_, agg_pk) = sdk.agg_keygen(agg_config, Some(AGG_PK_PATH)).unwrap();
    let (file_config, file_pk) = sdk.load_agg_pk_from_file(AGG_PK_PATH).unwrap();
    assert_agg_config_eq(&agg_config, &file_config);
    assert_agg_pk_eq(&agg_pk, &file_pk);
}

#[test]
fn test_sdk_vm_config_builder() {
    let sdk_vm_config = SdkVmConfig::builder()
        .system(
            SystemConfig::default()
                .with_max_segment_len(200)
                .with_continuations()
                .with_public_values(16),
        )
        .native(Default::default())
        .rv32i(Default::default())
        .build();
    let app_config = AppConfig {
        app_fri_params: standard_fri_params_with_100_bits_conjectured_security(1),
        app_vm_config: sdk_vm_config,
    };

    let (e2e_prover, dummy_internal_proof) = load_agg_pk_into_e2e_prover(app_config);
    let root_proof = e2e_prover.generate_e2e_proof(StdIn::default());

    static_verifier::test_static_verifier(
        &e2e_prover.agg_pk().root_verifier_pk,
        dummy_internal_proof,
        &root_proof,
    );
}

mod static_verifier {
    use ax_stark_sdk::{
        ax_stark_backend::prover::types::Proof,
        config::baby_bear_poseidon2_outer::BabyBearPoseidon2OuterConfig,
    };
    use axvm_native_compiler::prelude::Witness;
    use axvm_native_recursion::{halo2::wrapper::Halo2WrapperProvingKey, witness::Witnessable};
    use axvm_sdk::keygen::RootVerifierProvingKey;

    use crate::SC;

    pub(crate) fn test_static_verifier(
        root_verifier_pk: &RootVerifierProvingKey,
        dummy_internal_proof: Proof<SC>,
        root_proof: &Proof<BabyBearPoseidon2OuterConfig>,
    ) {
        // Here we intend to use a dummy root proof to generate a static verifier circuit in order
        // to test if the static verifier circuit can handle a different root proof.
        let dummy_root_proof = root_verifier_pk.generate_dummy_root_proof(dummy_internal_proof);
        let static_verifier = root_verifier_pk.keygen_static_verifier(24, dummy_root_proof);
        let mut witness = Witness::default();
        root_proof.write(&mut witness);
        // Here the proof is verified inside.
        // FIXME: explicitly verify the proof.
        let static_verifier_proof = static_verifier.prove(witness);
        let verifier_wrapper =
            Halo2WrapperProvingKey::keygen_auto_tune(static_verifier.generate_dummy_snark());
        assert_eq!(
            verifier_wrapper
                .pinning
                .metadata
                .config_params
                .num_advice_per_phase,
            vec![1]
        );
        let evm_verifier = verifier_wrapper.generate_evm_verifier();
        let evm_proof = verifier_wrapper.prove_for_evm(static_verifier_proof);
        Halo2WrapperProvingKey::evm_verify(evm_verifier, evm_proof);
    }
}
