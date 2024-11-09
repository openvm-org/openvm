use std::{borrow::Borrow, sync::Arc};

use ax_stark_sdk::{
    ax_stark_backend::{config::StarkGenericConfig, p3_field::AbstractField, prover::types::Proof},
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        baby_bear_poseidon2_outer::{BabyBearPoseidon2OuterConfig, BabyBearPoseidon2OuterEngine},
        fri_params::standard_fri_params_with_100_bits_conjectured_security,
    },
    engine::{StarkEngine, StarkFriEngine},
};
use axiom_vm::{
    commit::AppExecutionCommit,
    config::{AxiomVmConfig, AxiomVmProvingKey},
    verifier::{
        common::types::VmVerifierPvs,
        internal::types::InternalVmVerifierInput,
        leaf::types::{LeafVmVerifierInput, UserPublicValuesRootProof},
        root::types::{RootVmVerifierInput, RootVmVerifierPvs},
    },
};
use axvm_circuit::{
    arch::{
        hasher::poseidon2::vm_poseidon2_hasher, ExecutorName, SingleSegmentVmExecutor,
        VirtualMachine, VmConfig, VmExecutor, PUBLIC_VALUES_AIR_ID,
    },
    system::{
        memory::tree::public_values::UserPublicValuesProof,
        program::{trace::AxVmCommittedExe, ExecutionError},
    },
};
use axvm_native_compiler::{conversion::CompilerOptions, prelude::*};
use axvm_recursion::{hints::Hintable, types::InnerConfig};
use p3_baby_bear::BabyBear;

type SC = BabyBearPoseidon2Config;
type OuterSC = BabyBearPoseidon2OuterConfig;
type C = InnerConfig;
type F = BabyBear;
#[test]
fn test_1() {
    let fri_params = standard_fri_params_with_100_bits_conjectured_security(3);
    let axiom_vm_config = AxiomVmConfig {
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
    let max_num_user_public_values = axiom_vm_config.max_num_user_public_values;
    let axiom_vm_pk = AxiomVmProvingKey::keygen(axiom_vm_config);
    let app_engine = BabyBearPoseidon2Engine::new(axiom_vm_pk.app_fri_params);

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

    let app_vm = VmExecutor::new(axiom_vm_pk.app_vm_config.clone());
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
        .map(|proof_input| app_engine.prove(&axiom_vm_pk.app_vm_pk, proof_input))
        .collect();

    let last_proof = app_vm_seg_proofs.pop().unwrap();
    let leaf_vm = SingleSegmentVmExecutor::new(axiom_vm_pk.leaf_vm_config.clone());

    let run_leaf_verifier =
        |verifier_input: LeafVmVerifierInput<SC>| -> Result<Vec<F>, ExecutionError> {
            let runtime_pvs = leaf_vm.execute(
                axiom_vm_pk.leaf_committed_exe.exe.clone(),
                verifier_input.write_to_stream(),
            )?;
            let runtime_pvs: Vec<_> = runtime_pvs[..VmVerifierPvs::<u8>::width()]
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

    let leaf_vm = VirtualMachine::new(
        BabyBearPoseidon2Engine::new(axiom_vm_pk.leaf_fri_params),
        axiom_vm_pk.leaf_vm_config.clone(),
    );
    let internal_commit: [F; DIGEST_SIZE] = axiom_vm_pk
        .internal_committed_exe
        .committed_program
        .prover_data
        .commit
        .into();
    let prove_leaf_verifier = |verifier_input: LeafVmVerifierInput<SC>| -> Proof<SC> {
        let mut result = leaf_vm
            .execute_and_generate_with_cached_program(
                axiom_vm_pk.leaf_committed_exe.clone(),
                verifier_input.write_to_stream(),
            )
            .unwrap();
        let proof = result.per_segment.pop().unwrap();
        leaf_vm.prove_single(&axiom_vm_pk.leaf_vm_pk, proof)
    };
    let leaf_proofs = vec![
        prove_leaf_verifier(LeafVmVerifierInput {
            proofs: app_vm_seg_proofs.clone(),
            public_values_root_proof: None,
        }),
        prove_leaf_verifier(LeafVmVerifierInput {
            proofs: vec![last_proof.clone()],
            public_values_root_proof: Some(pv_root_proof.clone()),
        }),
    ];

    let internal_vm = VirtualMachine::new(
        BabyBearPoseidon2Engine::new(axiom_vm_pk.internal_fri_params),
        axiom_vm_pk.internal_vm_config.clone(),
    );
    let prove_internal_verifier = |verifier_input: InternalVmVerifierInput<SC>| -> Proof<SC> {
        let mut result = internal_vm
            .execute_and_generate_with_cached_program(
                axiom_vm_pk.internal_committed_exe.clone(),
                verifier_input.write(),
            )
            .unwrap();
        let proof = result.per_segment.pop().unwrap();
        internal_vm.prove_single(&axiom_vm_pk.internal_vm_pk, proof)
    };
    let internal_proofs = vec![prove_internal_verifier(InternalVmVerifierInput {
        self_program_commit: internal_commit,
        proofs: leaf_proofs.clone(),
    })];

    let root_agg_vm = VirtualMachine::new(
        BabyBearPoseidon2OuterEngine::new(axiom_vm_pk.root_fri_params),
        axiom_vm_pk.root_vm_config.clone(),
    );
    let prove_root_verifier = |verifier_input: RootVmVerifierInput<SC>| -> Proof<OuterSC> {
        let mut leaf_result = root_agg_vm
            .execute_and_generate_with_cached_program(
                axiom_vm_pk.root_committed_exe.clone(),
                verifier_input.write(),
            )
            .unwrap();
        let proof = leaf_result.per_segment.pop().unwrap();
        root_agg_vm.prove_single(&axiom_vm_pk.root_vm_pk, proof)
    };
    let app_exe_commit = AppExecutionCommit::compute(
        &axiom_vm_pk.app_vm_config,
        &committed_exe,
        &axiom_vm_pk.leaf_committed_exe,
    );

    let root_proof = prove_root_verifier(RootVmVerifierInput {
        proofs: internal_proofs.clone(),
        public_values: pv_proof.public_values,
    });
    let root_pvs = RootVmVerifierPvs::from_flatten(
        root_proof.per_air[PUBLIC_VALUES_AIR_ID]
            .public_values
            .clone(),
    );
    assert_eq!(root_pvs.exe_commit, app_exe_commit.exe_commit);
    assert_eq!(
        root_pvs.leaf_verifier_commit,
        app_exe_commit.leaf_vm_verifier_commit
    );
}
