use std::sync::Arc;

use ax_stark_sdk::{
    ax_stark_backend::{config::StarkGenericConfig, p3_field::AbstractField, prover::types::Proof},
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters},
    engine::{StarkEngine, StarkFriEngine},
};
use axvm_circuit::{
    arch::{
        instructions::{
            exe::AxVmExe, instruction::Instruction, program::Program, SystemOpcode::TERMINATE,
            UsizeOpcode,
        },
        SingleSegmentVmExecutor, VmConfig,
    },
    prover::{
        local::VmLocalProver, types::VmProvingKey, ContinuationVmProof, ContinuationVmProver,
        SingleSegmentVmProver,
    },
    system::program::trace::AxVmCommittedExe,
};
use axvm_native_compiler::ir::DIGEST_SIZE;
use axvm_recursion::hints::Hintable;

use crate::{
    verifier::{
        internal::types::InternalVmVerifierInput,
        leaf::{types::LeafVmVerifierInput, LeafVmVerifierConfig},
        root::types::RootVmVerifierInput,
    },
    F, SC,
};

pub(super) fn compute_root_proof_height(
    root_vm_config: VmConfig,
    root_exe: AxVmExe<F>,
    dummy_internal_proof: &Proof<SC>,
) -> Vec<usize> {
    let num_user_public_values = root_vm_config.num_public_values - 2 * DIGEST_SIZE;
    let root_input = RootVmVerifierInput {
        proofs: vec![dummy_internal_proof.clone()],
        public_values: vec![F::ZERO; num_user_public_values],
    };
    let vm = SingleSegmentVmExecutor::new(root_vm_config);
    let heights = vm.execute(root_exe, root_input.write()).unwrap().heights;
    heights.into_iter().map(|h| h.next_power_of_two()).collect()
}

pub(super) fn dummy_internal_proof(
    internal_vm_pk: VmProvingKey<SC>,
    internal_exe: Arc<AxVmCommittedExe<SC>>,
    leaf_proof: Proof<SC>,
) -> Proof<SC> {
    let mut internal_inputs = InternalVmVerifierInput::chunk_leaf_or_internal_proofs(
        internal_exe.get_program_commit().into(),
        &[leaf_proof],
        1,
    );
    let internal_input = internal_inputs.pop().unwrap();
    let internal_prover =
        VmLocalProver::<SC, BabyBearPoseidon2Engine>::new(internal_vm_pk, internal_exe);
    SingleSegmentVmProver::prove(&internal_prover, internal_input.write())
}

pub(super) fn dummy_leaf_proof(
    leaf_vm_pk: VmProvingKey<SC>,
    num_public_values: usize,
    app_fri_params: FriParameters,
) -> Proof<SC> {
    let app_vm_pk = dummy_riscv_app_vm_pk(num_public_values, app_fri_params);
    let app_proof = dummy_app_proof_impl(app_vm_pk.clone());
    dummy_leaf_proof_impl(leaf_vm_pk, &app_vm_pk, &app_proof)
}

fn dummy_leaf_proof_impl(
    leaf_vm_pk: VmProvingKey<SC>,
    app_vm_pk: &VmProvingKey<SC>,
    app_proof: &ContinuationVmProof<SC>,
) -> Proof<SC> {
    let leaf_program = LeafVmVerifierConfig {
        app_fri_params: app_vm_pk.fri_params,
        app_vm_config: app_vm_pk.vm_config.clone(),
        compiler_options: Default::default(),
    }
    .build_program(&app_vm_pk.vm_pk.get_vk());
    assert_eq!(
        app_proof.per_segment.len(),
        1,
        "Dummy proof should only have 1 segment"
    );
    let e = BabyBearPoseidon2Engine::new(app_vm_pk.fri_params);
    let leaf_exe = Arc::new(AxVmCommittedExe::<SC>::commit(
        leaf_program.into(),
        e.config.pcs(),
    ));
    let leaf_prover = VmLocalProver::<SC, BabyBearPoseidon2Engine>::new(leaf_vm_pk, leaf_exe);
    let mut leaf_inputs = LeafVmVerifierInput::chunk_continuation_vm_proof(app_proof, 1);
    let leaf_input = leaf_inputs.pop().unwrap();
    SingleSegmentVmProver::prove(&leaf_prover, leaf_input.write_to_stream())
}

fn dummy_riscv_app_vm_pk(num_public_values: usize, fri_params: FriParameters) -> VmProvingKey<SC> {
    let vm_config = VmConfig::rv32im().with_num_public_values(num_public_values);
    let vm_pk = vm_config.generate_pk(BabyBearPoseidon2Engine::new(fri_params).keygen_builder());
    VmProvingKey {
        fri_params,
        vm_config,
        vm_pk,
    }
}

fn dummy_app_proof_impl(mut app_vm_pk: VmProvingKey<SC>) -> ContinuationVmProof<SC> {
    // Enforce each AIR to have at least 1 row.
    app_vm_pk.vm_config.overridden_executor_heights = Some(
        app_vm_pk
            .vm_config
            .executors
            .iter()
            .map(|executor| (*executor, 1))
            .collect(),
    );
    let fri_params = app_vm_pk.fri_params;
    let app_prover = VmLocalProver::<SC, BabyBearPoseidon2Engine>::new(
        app_vm_pk,
        dummy_app_committed_exe(fri_params),
    );
    ContinuationVmProver::prove(&app_prover, vec![])
}

fn dummy_app_committed_exe(fri_params: FriParameters) -> Arc<AxVmCommittedExe<SC>> {
    let program = dummy_app_program();
    let e = BabyBearPoseidon2Engine::new(fri_params);
    Arc::new(AxVmCommittedExe::<SC>::commit(
        program.into(),
        e.config.pcs(),
    ))
}

fn dummy_app_program() -> Program<F> {
    let mut ret = Program::from_instructions(&[Instruction::from_isize(
        TERMINATE.with_default_offset(),
        0,
        0,
        0,
        0,
        0,
    )]);
    ret.max_num_public_values = 0;
    ret
}
