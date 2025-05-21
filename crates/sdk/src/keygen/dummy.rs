use std::sync::Arc;

use openvm_circuit::{
    arch::{
        instructions::{
            exe::VmExe, instruction::Instruction, program::Program, LocalOpcode,
            SystemOpcode::TERMINATE,
        },
        ContinuationVmProof, SingleSegmentVmExecutor, VirtualMachine, VmComplexTraceHeights,
        VmConfig, VmExecutor,
    },
    system::program::trace::VmCommittedExe,
    utils::next_power_of_two_or_zero,
};
use openvm_continuations::verifier::{
    internal::types::InternalVmVerifierInput,
    leaf::{types::LeafVmVerifierInput, LeafVmVerifierConfig},
    root::types::RootVmVerifierInput,
};
use openvm_native_circuit::NativeConfig;
use openvm_native_compiler::ir::DIGEST_SIZE;
use openvm_native_recursion::hints::Hintable;
use openvm_rv32im_circuit::Rv32ImConfig;
use openvm_stark_sdk::{
    config::{
        baby_bear_poseidon2::BabyBearPoseidon2Engine,
        fri_params::standard_fri_params_with_100_bits_conjectured_security, FriParameters,
    },
    engine::StarkFriEngine,
    openvm_stark_backend::{
        config::StarkGenericConfig, p3_field::FieldAlgebra, proof::Proof, Chip,
    },
};

use crate::{
    prover::vm::{
        local::VmLocalProver, types::VmProvingKey, ContinuationVmProver, SingleSegmentVmProver,
    },
    NonRootCommittedExe, F, SC,
};

/// Returns:
/// - trace heights ordered by AIR ID
/// - internal ordering of trace heights.
///
/// All trace heights are rounded to the next power of two (or 0 -> 0).
pub(super) fn compute_root_proof_heights(
    root_vm_config: NativeConfig,
    root_exe: VmExe<F>,
    dummy_internal_proof: &Proof<SC>,
) -> (Vec<usize>, VmComplexTraceHeights) {
    let num_user_public_values = root_vm_config.system.num_public_values - 2 * DIGEST_SIZE;
    let root_input = RootVmVerifierInput {
        proofs: vec![dummy_internal_proof.clone()],
        public_values: vec![F::ZERO; num_user_public_values],
    };
    let vm = SingleSegmentVmExecutor::new(root_vm_config);
    let res = vm
        .execute_and_compute_heights(root_exe, root_input.write())
        .unwrap();
    let air_heights: Vec<_> = res
        .air_heights
        .into_iter()
        .map(next_power_of_two_or_zero)
        .collect();
    let mut vm_heights = res.vm_heights;
    vm_heights.round_to_next_power_of_two_or_zero();
    (air_heights, vm_heights)
}

pub(super) fn dummy_internal_proof(
    internal_vm_pk: Arc<VmProvingKey<SC, NativeConfig>>,
    internal_exe: Arc<NonRootCommittedExe>,
    leaf_proof: Proof<SC>,
) -> Proof<SC> {
    let mut internal_inputs = InternalVmVerifierInput::chunk_leaf_or_internal_proofs(
        internal_exe.get_program_commit().into(),
        &[leaf_proof],
        1,
    );
    let internal_input = internal_inputs.pop().unwrap();
    let internal_prover = VmLocalProver::<SC, NativeConfig, BabyBearPoseidon2Engine>::new(
        internal_vm_pk,
        internal_exe,
    );
    SingleSegmentVmProver::prove(&internal_prover, internal_input.write())
}

pub(super) fn dummy_internal_proof_riscv_app_vm(
    leaf_vm_pk: Arc<VmProvingKey<SC, NativeConfig>>,
    internal_vm_pk: Arc<VmProvingKey<SC, NativeConfig>>,
    internal_exe: Arc<NonRootCommittedExe>,
    num_public_values: usize,
) -> Proof<SC> {
    let fri_params = standard_fri_params_with_100_bits_conjectured_security(1);
    let leaf_proof = dummy_leaf_proof_riscv_app_vm(leaf_vm_pk, num_public_values, fri_params);
    dummy_internal_proof(internal_vm_pk, internal_exe, leaf_proof)
}

#[allow(dead_code)]
pub fn dummy_leaf_proof<VC: VmConfig<F>>(
    leaf_vm_pk: Arc<VmProvingKey<SC, NativeConfig>>,
    app_vm_pk: Arc<VmProvingKey<SC, VC>>,
    overridden_heights: Option<VmComplexTraceHeights>,
) -> Proof<SC>
where
    VC::Executor: Chip<SC>,
    VC::Periphery: Chip<SC>,
{
    let app_proof = dummy_app_proof_impl(app_vm_pk.clone(), overridden_heights);
    dummy_leaf_proof_impl(leaf_vm_pk, app_vm_pk, &app_proof)
}

pub(super) fn dummy_leaf_proof_riscv_app_vm(
    leaf_vm_pk: Arc<VmProvingKey<SC, NativeConfig>>,
    num_public_values: usize,
    app_fri_params: FriParameters,
) -> Proof<SC> {
    let app_vm_pk = Arc::new(dummy_riscv_app_vm_pk(num_public_values, app_fri_params));
    let app_proof = dummy_app_proof_impl(app_vm_pk.clone(), None);
    dummy_leaf_proof_impl(leaf_vm_pk, app_vm_pk, &app_proof)
}

fn dummy_leaf_proof_impl<VC: VmConfig<F>>(
    leaf_vm_pk: Arc<VmProvingKey<SC, NativeConfig>>,
    app_vm_pk: Arc<VmProvingKey<SC, VC>>,
    app_proof: &ContinuationVmProof<SC>,
) -> Proof<SC> {
    let leaf_program = LeafVmVerifierConfig {
        app_fri_params: app_vm_pk.fri_params,
        app_system_config: app_vm_pk.vm_config.system().clone(),
        compiler_options: Default::default(),
    }
    .build_program(&app_vm_pk.vm_pk.get_vk());
    assert_eq!(
        app_proof.per_segment.len(),
        1,
        "Dummy proof should only have 1 segment"
    );
    let e = BabyBearPoseidon2Engine::new(leaf_vm_pk.fri_params);
    let leaf_exe = Arc::new(VmCommittedExe::<SC>::commit(
        leaf_program.into(),
        e.config.pcs(),
    ));
    let leaf_prover =
        VmLocalProver::<SC, NativeConfig, BabyBearPoseidon2Engine>::new(leaf_vm_pk, leaf_exe);
    let mut leaf_inputs = LeafVmVerifierInput::chunk_continuation_vm_proof(app_proof, 1);
    let leaf_input = leaf_inputs.pop().unwrap();
    SingleSegmentVmProver::prove(&leaf_prover, leaf_input.write_to_stream())
}

fn dummy_riscv_app_vm_pk(
    num_public_values: usize,
    fri_params: FriParameters,
) -> VmProvingKey<SC, Rv32ImConfig> {
    let vm_config = Rv32ImConfig::with_public_values(num_public_values);
    let vm = VirtualMachine::new(BabyBearPoseidon2Engine::new(fri_params), vm_config.clone());
    let vm_pk = vm.keygen();
    VmProvingKey {
        fri_params,
        vm_config,
        vm_pk,
    }
}

fn dummy_app_proof_impl<VC: VmConfig<F>>(
    app_vm_pk: Arc<VmProvingKey<SC, VC>>,
    overridden_heights: Option<VmComplexTraceHeights>,
) -> ContinuationVmProof<SC>
where
    VC::Executor: Chip<SC>,
    VC::Periphery: Chip<SC>,
{
    let fri_params = app_vm_pk.fri_params;
    let dummy_exe = dummy_app_committed_exe(fri_params);
    // Enforce each AIR to have at least 1 row.
    let overridden_heights = if let Some(overridden_heights) = overridden_heights {
        overridden_heights
    } else {
        // We first execute once to get the trace heights from dummy_exe, then pad to powers of 2
        // (forcing trace height 0 to 1)
        let executor = VmExecutor::new(app_vm_pk.vm_config.clone());
        let mut results = executor
            .execute_segments(dummy_exe.exe.clone(), vec![])
            .unwrap();
        // ASSUMPTION: the dummy exe has only 1 segment
        assert_eq!(results.len(), 1, "dummy exe should have only 1 segment");
        let mut result = results.pop().unwrap();
        result.chip_complex.finalize_memory();
        let mut vm_heights = result.chip_complex.get_internal_trace_heights();
        vm_heights.round_to_next_power_of_two();
        vm_heights
    };
    // For the dummy proof, we must override the trace heights.
    let app_prover =
        VmLocalProver::<SC, VC, BabyBearPoseidon2Engine>::new_with_overridden_trace_heights(
            app_vm_pk,
            dummy_exe,
            Some(overridden_heights),
        );
    ContinuationVmProver::prove(&app_prover, vec![])
}

fn dummy_app_committed_exe(fri_params: FriParameters) -> Arc<NonRootCommittedExe> {
    let program = dummy_app_program();
    let e = BabyBearPoseidon2Engine::new(fri_params);
    Arc::new(VmCommittedExe::<SC>::commit(program.into(), e.config.pcs()))
}

fn dummy_app_program() -> Program<F> {
    Program::from_instructions(&[Instruction::from_isize(
        TERMINATE.global_opcode(),
        0,
        0,
        0,
        0,
        0,
    )])
}
