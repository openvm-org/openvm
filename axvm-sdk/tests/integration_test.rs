use std::{borrow::Borrow, sync::Arc};

use ax_stark_sdk::{
    ax_stark_backend::{config::StarkGenericConfig, p3_field::AbstractField},
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        fri_params::standard_fri_params_with_100_bits_conjectured_security,
    },
    engine::{StarkEngine, StarkFriEngine},
};
use axvm_circuit::{
    arch::{
        hasher::poseidon2::vm_poseidon2_hasher, ExecutorName, SingleSegmentVmExecutor, VmConfig,
        VmExecutor,
    },
    prover::{local::VmLocalProver, SingleSegmentVmProver},
    system::{
        memory::tree::public_values::UserPublicValuesProof,
        program::{trace::AxVmCommittedExe, ExecutionError},
    },
};
use axvm_native_compiler::{conversion::CompilerOptions, prelude::*};
use axvm_recursion::{hints::Hintable, types::InnerConfig};
use axvm_sdk::{
    commit::AppExecutionCommit,
    config::AxVmSdkConfig,
    keygen::AxVmSdkProvingKey,
    prover::RootVerifierLocalProver,
    verifier::{
        common::types::VmVerifierPvs,
        internal::types::InternalVmVerifierInput,
        leaf::types::{LeafVmVerifierInput, UserPublicValuesRootProof},
        root::types::{RootVmVerifierInput, RootVmVerifierPvs},
    },
};
use p3_baby_bear::BabyBear;

type SC = BabyBearPoseidon2Config;
type C = InnerConfig;
type F = BabyBear;
#[test]
fn test_1() {
    let fri_params = standard_fri_params_with_100_bits_conjectured_security(3);
    let axvm_sdk_config = AxVmSdkConfig {
        max_num_user_public_values: 16,
        app_fri_params: fri_params,
        leaf_fri_params: fri_params,
        internal_fri_params: fri_params,
        root_fri_params: fri_params,
        app_vm_config: VmConfig {
            max_segment_len: 200,
            continuation_enabled: true,
            num_public_values: 16,
            ..Default::default()
        }
        .add_executor(ExecutorName::BranchEqual)
        .add_executor(ExecutorName::Jal)
        .add_executor(ExecutorName::LoadStore)
        .add_executor(ExecutorName::FieldArithmetic),
        compiler_options: CompilerOptions {
            enable_cycle_tracker: true,
            compile_prints: true,
            ..Default::default()
        },
    };
    let max_num_user_public_values = axvm_sdk_config.max_num_user_public_values;
    #[allow(unused_variables)]
    let (axvm_sdk_pk, dummy_internal_proof) = AxVmSdkProvingKey::keygen_impl(axvm_sdk_config);
    let app_engine = BabyBearPoseidon2Engine::new(axvm_sdk_pk.app_vm_pk.fri_params);

    let mut program = {
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
    program.max_num_public_values = 16;
    let committed_exe = Arc::new(AxVmCommittedExe::<SC>::commit(
        program.into(),
        app_engine.config.pcs(),
    ));

    let expected_program_commit: [F; DIGEST_SIZE] =
        committed_exe.committed_program.prover_data.commit.into();

    let app_vm = VmExecutor::new(axvm_sdk_pk.app_vm_pk.vm_config.clone());
    let app_vm_result = app_vm
        .execute_and_generate_with_cached_program(committed_exe.clone(), vec![])
        .unwrap();
    assert!(app_vm_result.per_segment.len() > 2);

    let pv_proof = UserPublicValuesProof::compute(
        app_vm.config.memory_config.memory_dimensions(),
        max_num_user_public_values,
        &vm_poseidon2_hasher(),
        app_vm_result.final_memory.as_ref().unwrap(),
    );
    let pv_root_proof = UserPublicValuesRootProof::extract(&pv_proof);
    let expected_pv_commit = pv_root_proof.public_values_commit;
    let mut app_vm_seg_proofs: Vec<_> = app_vm_result
        .per_segment
        .into_iter()
        .map(|proof_input| app_engine.prove(&axvm_sdk_pk.app_vm_pk.vm_pk, proof_input))
        .collect();

    let last_proof = app_vm_seg_proofs.pop().unwrap();
    let leaf_vm = SingleSegmentVmExecutor::new(axvm_sdk_pk.leaf_vm_pk.vm_config.clone());

    let run_leaf_verifier =
        |verifier_input: LeafVmVerifierInput<SC>| -> Result<Vec<F>, ExecutionError> {
            let exe_result = leaf_vm.execute(
                axvm_sdk_pk.leaf_committed_exe.exe.clone(),
                verifier_input.write_to_stream(),
            )?;
            let runtime_pvs: Vec<_> = exe_result
                .public_values
                .iter()
                .map(|v| v.unwrap())
                .collect();
            Ok(runtime_pvs)
        };

    // Verify all segments except the last one.
    let (first_seg_final_pc, first_seg_final_mem_root) = {
        let runtime_pvs = run_leaf_verifier(LeafVmVerifierInput {
            proofs: app_vm_seg_proofs.clone(),
            public_values_root_proof: None,
        })
        .expect("failed to verify the first segment");
        let leaf_vm_pvs: &VmVerifierPvs<F> = runtime_pvs.as_slice().borrow();

        assert_eq!(leaf_vm_pvs.app_commit, expected_program_commit);
        assert_eq!(leaf_vm_pvs.connector.is_terminate, F::ZERO);
        assert_eq!(leaf_vm_pvs.connector.initial_pc, F::ZERO);
        (
            leaf_vm_pvs.connector.final_pc,
            leaf_vm_pvs.memory.final_root,
        )
    };
    // Verify the last segment with the correct public values root proof.
    {
        let runtime_pvs = run_leaf_verifier(LeafVmVerifierInput {
            proofs: vec![last_proof.clone()],
            public_values_root_proof: Some(pv_root_proof.clone()),
        })
        .expect("failed to verify the second segment");
        let leaf_vm_pvs: &VmVerifierPvs<F> = runtime_pvs.as_slice().borrow();
        assert_eq!(leaf_vm_pvs.app_commit, expected_program_commit);
        assert_eq!(leaf_vm_pvs.connector.initial_pc, first_seg_final_pc);
        assert_eq!(leaf_vm_pvs.connector.is_terminate, F::ONE);
        assert_eq!(leaf_vm_pvs.connector.exit_code, F::ZERO);
        assert_eq!(leaf_vm_pvs.memory.initial_root, first_seg_final_mem_root);
        assert_eq!(leaf_vm_pvs.public_values_commit, expected_pv_commit);
    }
    // Failure: The public value root proof has a wrong public values commit.
    {
        let mut wrong_pv_root_proof = pv_root_proof.clone();
        wrong_pv_root_proof.public_values_commit[0] += F::ONE;
        let execution_result = run_leaf_verifier(LeafVmVerifierInput {
            proofs: vec![last_proof.clone()],
            public_values_root_proof: Some(wrong_pv_root_proof),
        });
        match execution_result.err().unwrap() {
            ExecutionError::Fail(_) => {}
            _ => panic!("Expected execution to fail"),
        }
    }
    // Failure: The public value root proof has a wrong path proof.
    {
        let mut wrong_pv_root_proof = pv_root_proof.clone();
        wrong_pv_root_proof.sibling_hashes[0][0] += F::ONE;
        let execution_result = run_leaf_verifier(LeafVmVerifierInput {
            proofs: vec![last_proof.clone()],
            public_values_root_proof: Some(wrong_pv_root_proof),
        });
        match execution_result.err().unwrap() {
            ExecutionError::Fail(_) => {}
            _ => panic!("Expected execution to fail"),
        }
    }

    let leaf_prover = VmLocalProver::<SC, BabyBearPoseidon2Engine>::new(
        axvm_sdk_pk.leaf_vm_pk.clone(),
        axvm_sdk_pk.leaf_committed_exe.clone(),
    );
    let internal_commit: [F; DIGEST_SIZE] = axvm_sdk_pk
        .internal_committed_exe
        .get_program_commit()
        .into();
    let leaf_proofs = vec![
        SingleSegmentVmProver::prove(
            &leaf_prover,
            LeafVmVerifierInput {
                proofs: app_vm_seg_proofs.clone(),
                public_values_root_proof: None,
            }
            .write_to_stream(),
        ),
        SingleSegmentVmProver::prove(
            &leaf_prover,
            LeafVmVerifierInput {
                proofs: vec![last_proof.clone()],
                public_values_root_proof: Some(pv_root_proof.clone()),
            }
            .write_to_stream(),
        ),
    ];

    let internal_prover = VmLocalProver::<SC, BabyBearPoseidon2Engine>::new(
        axvm_sdk_pk.internal_vm_pk.clone(),
        axvm_sdk_pk.internal_committed_exe.clone(),
    );
    let internal_proofs = vec![SingleSegmentVmProver::prove(
        &internal_prover,
        InternalVmVerifierInput {
            self_program_commit: internal_commit,
            proofs: leaf_proofs.clone(),
        }
        .write(),
    )];

    let root_prover = RootVerifierLocalProver::new(axvm_sdk_pk.root_verifier_pk.clone());
    let app_exe_commit = AppExecutionCommit::compute(
        &axvm_sdk_pk.app_vm_pk.vm_config,
        &committed_exe,
        &axvm_sdk_pk.leaf_committed_exe,
    );

    let root_proof = SingleSegmentVmProver::prove(
        &root_prover,
        RootVmVerifierInput {
            proofs: internal_proofs.clone(),
            public_values: pv_proof.public_values,
        }
        .write(),
    );
    let air_id_perm = axvm_sdk_pk.root_verifier_pk.air_id_permutation();
    let special_air_ids = air_id_perm.get_special_air_ids();
    let root_pvs = RootVmVerifierPvs::from_flatten(
        root_proof.per_air[special_air_ids.public_values_air_id]
            .public_values
            .clone(),
    );
    assert_eq!(root_pvs.exe_commit, app_exe_commit.exe_commit);
    assert_eq!(
        root_pvs.leaf_verifier_commit,
        app_exe_commit.leaf_vm_verifier_commit
    );
    #[cfg(feature = "static-verifier")]
    static_verifier::test_static_verifier(
        &axvm_sdk_pk.root_verifier_pk,
        dummy_internal_proof,
        &root_proof,
    );
}
#[cfg(feature = "static-verifier")]
mod static_verifier {
    use ax_stark_sdk::{
        ax_stark_backend::prover::types::Proof,
        config::baby_bear_poseidon2_outer::BabyBearPoseidon2OuterConfig,
    };
    use axvm_native_compiler::prelude::Witness;
    use axvm_recursion::witness::Witnessable;
    use axvm_sdk::keygen::RootVerifierProvingKey;

    use crate::SC;

    pub(crate) fn test_static_verifier(
        root_verifier_pk: &RootVerifierProvingKey,
        dummy_internal_proof: Proof<SC>,
        root_proot: &Proof<BabyBearPoseidon2OuterConfig>,
    ) {
        let static_verifier = root_verifier_pk.keygen_static_verifier(23, dummy_internal_proof);
        let mut witness = Witness::default();
        root_proot.write(&mut witness);
        // Here the proof is verified inside.
        // FIXME: explicitly verify the proof.
        static_verifier.prove(witness);
    }
}
