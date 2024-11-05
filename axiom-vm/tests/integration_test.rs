use std::{borrow::Borrow, sync::Arc};

use ax_stark_sdk::{
    ax_stark_backend::{config::StarkGenericConfig, p3_field::AbstractField},
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        fri_params::standard_fri_params_with_100_bits_conjectured_security,
    },
    engine::{StarkEngine, StarkFriEngine},
};
use axiom_vm::{
    config::{AxiomVmConfig, AxiomVmProvingKey},
    verifier::leaf::types::{LeafVmVerifierInput, LeafVmVerifierPvs, UserPublicValuesRootProof},
};
use axvm_circuit::{
    arch::{
        hasher::poseidon2::vm_poseidon2_hasher, ExecutorName, SingleSegmentVmExecutor, VmConfig,
        VmExecutor,
    },
    system::{
        memory::tree::public_values::compute_user_public_values_proof,
        program::{trace::AxVmCommittedExe, ExecutionError},
    },
};
use axvm_native_compiler::{conversion::CompilerOptions, prelude::*};
use axvm_recursion::types::InnerConfig;
use p3_baby_bear::BabyBear;

type SC = BabyBearPoseidon2Config;
type C = InnerConfig;
type F = BabyBear;
#[test]
fn test_1() {
    let axiom_vm_config = AxiomVmConfig {
        poseidon2_max_constraint_degree: 7,
        max_num_user_public_values: 16,
        fri_params: standard_fri_params_with_100_bits_conjectured_security(3),
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
    let max_num_user_public_values = axiom_vm_config.max_num_user_public_values;
    let axiom_vm_pk = AxiomVmProvingKey::keygen(axiom_vm_config);
    let engine = BabyBearPoseidon2Engine::new(axiom_vm_pk.fri_params);

    let program = {
        let n = 200;
        let mut builder = Builder::<C>::default();
        let a: Felt<F> = builder.eval(F::zero());
        let b: Felt<F> = builder.eval(F::one());
        let c: Felt<F> = builder.uninit();
        builder.range(0, n).for_each(|_, builder| {
            builder.assign(&c, a + b);
            builder.assign(&a, b);
            builder.assign(&b, c);
        });
        builder.halt();
        builder.compile_isa()
    };
    let committed_exe = Arc::new(AxVmCommittedExe::<SC>::commit(
        program.into(),
        engine.config.pcs(),
    ));

    let expected_program_commit: [F; DIGEST_SIZE] =
        committed_exe.committed_program.prover_data.commit.into();

    let app_vm = VmExecutor::new(axiom_vm_pk.app_vm_config.clone());
    let app_vm_result = app_vm
        .execute_and_generate_with_cached_program(committed_exe, vec![])
        .unwrap();
    assert!(app_vm_result.per_segment.len() > 2);

    let pv_proof = compute_user_public_values_proof(
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
        .map(|proof_input| engine.prove(&axiom_vm_pk.app_vm_pk, proof_input))
        .collect();

    let leaf_vm = SingleSegmentVmExecutor::new(axiom_vm_pk.leaf_vm_config);
    let last_proof = app_vm_seg_proofs.pop().unwrap();

    let run_leaf_verifier =
        |verifier_input: LeafVmVerifierInput<SC>| -> Result<Vec<F>, ExecutionError> {
            let runtime_pvs = leaf_vm.execute(
                axiom_vm_pk.committed_leaf_program.exe.clone(),
                verifier_input.write_to_stream(),
            )?;
            let runtime_pvs: Vec<_> = runtime_pvs.into_iter().map(|v| v.unwrap()).collect();
            Ok(runtime_pvs)
        };

    // Verify all segments except the last one.
    let (first_seg_final_pc, first_seg_final_mem_root) = {
        let runtime_pvs = run_leaf_verifier(LeafVmVerifierInput {
            proofs: app_vm_seg_proofs,
            public_values_root_proof: None,
        })
        .expect("failed to verify the first segment");
        let leaf_vm_pvs: &LeafVmVerifierPvs<F> = runtime_pvs.as_slice().borrow();

        assert_eq!(leaf_vm_pvs.app_commit, expected_program_commit);
        assert_eq!(leaf_vm_pvs.connector.is_terminate, F::zero());
        assert_eq!(leaf_vm_pvs.connector.initial_pc, F::zero());
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
        let leaf_vm_pvs: &LeafVmVerifierPvs<F> = runtime_pvs.as_slice().borrow();
        assert_eq!(leaf_vm_pvs.app_commit, expected_program_commit);
        assert_eq!(leaf_vm_pvs.connector.initial_pc, first_seg_final_pc);
        assert_eq!(leaf_vm_pvs.connector.is_terminate, F::one());
        assert_eq!(leaf_vm_pvs.connector.exit_code, F::zero());
        assert_eq!(leaf_vm_pvs.memory.initial_root, first_seg_final_mem_root);
        assert_eq!(leaf_vm_pvs.public_values_commit, expected_pv_commit);
    }
    // Failure: The public value root proof has a wrong public values commit.
    {
        let mut wrong_pv_root_proof = pv_root_proof.clone();
        wrong_pv_root_proof.public_values_commit[0] += F::one();
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
        wrong_pv_root_proof.sibling_hashes[0][0] += F::one();
        let execution_result = run_leaf_verifier(LeafVmVerifierInput {
            proofs: vec![last_proof],
            public_values_root_proof: Some(wrong_pv_root_proof),
        });
        match execution_result.err().unwrap() {
            ExecutionError::Fail(_) => {}
            _ => panic!("Expected execution to fail"),
        }
    }
}
