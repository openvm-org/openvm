use std::sync::Arc;

use openvm_circuit::{
    arch::{
        instructions::{
            exe::VmExe, instruction::Instruction, program::Program, LocalOpcode,
            SystemOpcode::TERMINATE,
        },
        ContinuationVmProof, Executor, MatrixRecordArena, MeteredExecutor,
        PreflightExecutionOutput, PreflightExecutor, SingleSegmentVmProver, SystemConfig,
        VirtualMachine, VirtualMachineError, VmBuilder, VmExecutionConfig, PUBLIC_VALUES_AIR_ID,
    },
    system::program::trace::VmCommittedExe,
    utils::next_power_of_two_or_zero,
};
use openvm_continuations::verifier::{
    internal::types::InternalVmVerifierInput,
    leaf::{types::LeafVmVerifierInput, LeafVmVerifierConfig},
    root::types::RootVmVerifierInput,
};
use openvm_native_circuit::{NativeConfig, NativeCpuBuilder, NATIVE_MAX_TRACE_HEIGHTS};
use openvm_native_compiler::ir::DIGEST_SIZE;
use openvm_native_recursion::hints::Hintable;
use openvm_rv32im_circuit::{Rv32ImConfig, Rv32ImCpuBuilder};
use openvm_stark_backend::{
    p3_matrix::dense::RowMajorMatrix,
    prover::{
        cpu::CpuBackend,
        types::{AirProvingContext, ProvingContext},
    },
};
use openvm_stark_sdk::{
    config::{
        baby_bear_poseidon2::BabyBearPoseidon2Engine,
        baby_bear_poseidon2_root::{BabyBearPoseidon2RootConfig, BabyBearPoseidon2RootEngine},
        fri_params::standard_fri_params_with_100_bits_conjectured_security,
        FriParameters,
    },
    engine::StarkFriEngine,
    openvm_stark_backend::{p3_field::FieldAlgebra, proof::Proof},
};

use crate::{
    prover::vm::{new_local_prover, types::VmProvingKey},
    F, SC,
};

/// Given a dummy internal proof, which is the input to the root verifier circuit, we will run
/// tracegen on the root verifier circuit to determine the trace heights. These trace heights will
/// become the fixed trace heights that we **force** the root verifier circuit's trace matrices to
/// have.
///
/// Returns:
/// - trace heights ordered by AIR ID
///
/// All trace heights are rounded to the next power of two (or 0 -> 0).
pub(super) fn compute_root_proof_heights(
    root_vm: &mut VirtualMachine<BabyBearPoseidon2RootEngine, NativeCpuBuilder>,
    root_committed_exe: &VmCommittedExe<BabyBearPoseidon2RootConfig>,
    dummy_internal_proof: &Proof<SC>,
) -> Result<Vec<u32>, VirtualMachineError> {
    let num_public_values = root_vm.config().as_ref().num_public_values;
    let num_user_public_values = num_public_values - 2 * DIGEST_SIZE;
    let root_input = RootVmVerifierInput {
        proofs: vec![dummy_internal_proof.clone()],
        public_values: vec![F::ZERO; num_user_public_values],
    };
    // The following is the same as impl SingleSegmentVmProver for VmLocalProver except we stop
    // after tracegen:
    let mut trace_heights = NATIVE_MAX_TRACE_HEIGHTS.to_vec();
    trace_heights[PUBLIC_VALUES_AIR_ID] = num_public_values as u32;
    let state = root_vm.create_initial_state(&root_committed_exe.exe, root_input.write());
    let cached_program_trace = root_vm.transport_committed_exe_to_device(root_committed_exe);
    root_vm.load_program(cached_program_trace);
    root_vm.transport_init_memory_to_device(&state.memory);
    let mut preflight_interpreter = root_vm.preflight_interpreter(&root_committed_exe.exe)?;
    let PreflightExecutionOutput {
        system_records,
        record_arenas,
        ..
    } = root_vm.execute_preflight(&mut preflight_interpreter, state, None, &trace_heights)?;
    let ctx = root_vm.generate_proving_ctx(system_records, record_arenas)?;
    let air_heights = ctx
        .into_iter()
        .map(|(_, air_ctx)| {
            next_power_of_two_or_zero(air_ctx.main_trace_height())
                .try_into()
                .unwrap()
        })
        .collect();
    Ok(air_heights)
}

pub(super) fn dummy_internal_proof(
    internal_vm_pk: Arc<VmProvingKey<SC, NativeConfig>>,
    internal_committed_exe: Arc<VmCommittedExe<SC>>,
    leaf_proof: Proof<SC>,
) -> Result<Proof<SC>, VirtualMachineError> {
    let mut internal_inputs = InternalVmVerifierInput::chunk_leaf_or_internal_proofs(
        internal_committed_exe.get_program_commit().into(),
        &[leaf_proof],
        1,
    );
    let internal_input = internal_inputs.pop().unwrap();
    let mut internal_prover = new_local_prover::<BabyBearPoseidon2Engine, _>(
        NativeCpuBuilder,
        &internal_vm_pk,
        internal_committed_exe.exe.clone(),
    )?;
    SingleSegmentVmProver::prove(
        &mut internal_prover,
        internal_input.write(),
        NATIVE_MAX_TRACE_HEIGHTS,
    )
}

pub(super) fn dummy_internal_proof_riscv_app_vm(
    leaf_vm_pk: Arc<VmProvingKey<SC, NativeConfig>>,
    internal_vm_pk: Arc<VmProvingKey<SC, NativeConfig>>,
    internal_exe: Arc<VmCommittedExe<SC>>,
    num_public_values: usize,
) -> Result<Proof<SC>, VirtualMachineError> {
    let fri_params = standard_fri_params_with_100_bits_conjectured_security(1);
    let leaf_proof = dummy_leaf_proof_riscv_app_vm(leaf_vm_pk, num_public_values, fri_params)?;
    dummy_internal_proof(internal_vm_pk, internal_exe, leaf_proof)
}

pub(super) fn dummy_leaf_proof_riscv_app_vm(
    leaf_vm_pk: Arc<VmProvingKey<SC, NativeConfig>>,
    num_public_values: usize,
    app_fri_params: FriParameters,
) -> Result<Proof<SC>, VirtualMachineError> {
    let app_vm_pk = Arc::new(dummy_riscv_app_vm_pk(num_public_values, app_fri_params)?);
    let app_proof = dummy_app_proof(Rv32ImCpuBuilder, app_vm_pk.clone())?;
    dummy_leaf_proof(leaf_vm_pk, app_vm_pk, &app_proof)
}

fn dummy_leaf_proof<VC>(
    leaf_vm_pk: Arc<VmProvingKey<SC, NativeConfig>>,
    app_vm_pk: Arc<VmProvingKey<SC, VC>>,
    app_proof: &ContinuationVmProof<SC>,
) -> Result<Proof<SC>, VirtualMachineError>
where
    VC: AsRef<SystemConfig>,
{
    let leaf_program = LeafVmVerifierConfig {
        app_fri_params: app_vm_pk.fri_params,
        app_system_config: app_vm_pk.vm_config.as_ref().clone(),
        compiler_options: Default::default(),
    }
    .build_program(&app_vm_pk.vm_pk.get_vk());
    assert_eq!(
        app_proof.per_segment.len(),
        1,
        "Dummy proof should only have 1 segment"
    );
    let leaf_exe = Arc::new(VmExe::new(leaf_program));
    let mut leaf_prover =
        new_local_prover::<BabyBearPoseidon2Engine, _>(NativeCpuBuilder, &leaf_vm_pk, leaf_exe)?;
    let mut leaf_inputs = LeafVmVerifierInput::chunk_continuation_vm_proof(app_proof, 1);
    let leaf_input = leaf_inputs.pop().unwrap();
    SingleSegmentVmProver::prove(
        &mut leaf_prover,
        leaf_input.write_to_stream(),
        NATIVE_MAX_TRACE_HEIGHTS,
    )
}

fn dummy_riscv_app_vm_pk(
    num_public_values: usize,
    fri_params: FriParameters,
) -> Result<VmProvingKey<SC, Rv32ImConfig>, VirtualMachineError> {
    let vm_config = Rv32ImConfig::with_public_values(num_public_values);
    let (_, vm_pk) = VirtualMachine::new_with_keygen(
        BabyBearPoseidon2Engine::new(fri_params),
        Rv32ImCpuBuilder,
        vm_config.clone(),
    )?;
    Ok(VmProvingKey {
        fri_params,
        vm_config,
        vm_pk,
    })
}

fn dummy_app_proof<VB, VC>(
    app_vm_builder: VB,
    app_vm_pk: Arc<VmProvingKey<SC, VC>>,
) -> Result<ContinuationVmProof<SC>, VirtualMachineError>
where
    VB: VmBuilder<BabyBearPoseidon2Engine, VmConfig = VC, RecordArena = MatrixRecordArena<F>>,
    VC: VmExecutionConfig<F>,
    <VC as VmExecutionConfig<F>>::Executor: Executor<F> + MeteredExecutor<F> + PreflightExecutor<F>,
{
    let dummy_exe = Arc::new(VmExe::new(dummy_app_program()));
    let mut app_prover =
        new_local_prover::<BabyBearPoseidon2Engine, VB>(app_vm_builder, &app_vm_pk, dummy_exe)?;
    // Force all AIRs to have non-empty trace matrices (height 0 -> height 1)
    let modify_ctx = |_seg_idx: usize, ctx: &mut ProvingContext<CpuBackend<SC>>| {
        for (i, pk) in app_vm_pk.vm_pk.per_air.iter().enumerate() {
            let width = pk.vk.params.width.common_main;
            if ctx.per_air[i].0 != i {
                let dummy_trace = RowMajorMatrix::new_row(F::zero_vec(width));
                let dummy_ctx = AirProvingContext::simple_no_pis(Arc::new(dummy_trace));
                ctx.per_air.insert(i, (i, dummy_ctx));
            }
        }
    };
    let dummy_proof = app_prover.prove_continuations(vec![], modify_ctx)?;
    Ok(dummy_proof)
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
