use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller, EmptyAdapterCoreLayout,
        ExecutionError, MinimalInstruction, PreflightExecutor, RecordArena, TraceFiller,
        VmAdapterInterface, VmCoreAir, VmStateMut,
    },
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_deferral_transpiler::DeferralOpcode;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing},
    BaseAirWithPublicValues,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
use p3_field::PrimeField32;

// ========================= AIR ==============================

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct DeferralSetupCoreCols<T> {
    pub is_valid: T,
}

#[derive(Copy, Clone, Debug, derive_new::new)]
pub struct DeferralSetupCoreAir<F> {
    expected_def_vks_commit: [F; DIGEST_SIZE],
}

impl<F: Sync> BaseAir<F> for DeferralSetupCoreAir<F> {
    fn width(&self) -> usize {
        DeferralSetupCoreCols::<F>::width()
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for DeferralSetupCoreAir<F> {}

impl<AB, I> VmCoreAir<AB, I> for DeferralSetupCoreAir<AB::F>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; 0]; 0]>,
    I::Writes: From<[[AB::Expr; DIGEST_SIZE]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &DeferralSetupCoreCols<_> = local_core.borrow();
        builder.assert_bool(cols.is_valid);

        AdapterAirContext {
            to_pc: None,
            reads: [].into(),
            writes: [self.expected_def_vks_commit.map(Into::into)].into(),
            instruction: MinimalInstruction {
                is_valid: cols.is_valid.into(),
                opcode: AB::Expr::from_usize(DeferralOpcode::SETUP.global_opcode_usize()),
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        DeferralOpcode::CLASS_OFFSET
    }
}

// ========================= EXECUTION + TRACEGEN ==============================

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct DeferralSetupCoreExecutor<F, A> {
    pub(in crate::setup) adapter: A,
    pub(in crate::setup) expected_def_vks_commit: [F; DIGEST_SIZE],
}

#[derive(Clone, Debug, derive_new::new)]
pub struct DeferralSetupCoreFiller<A> {
    adapter: A,
}

impl<F, A, RA> PreflightExecutor<F, RA> for DeferralSetupCoreExecutor<F, A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceExecutor<F, ReadData = (), WriteData = [F; DIGEST_SIZE]>,
    for<'buf> RA: RecordArena<'buf, EmptyAdapterCoreLayout<F, A>, (A::RecordMut<'buf>, ())>,
{
    fn get_opcode_name(&self, _opcode: usize) -> String {
        format!("{:?}", DeferralOpcode::SETUP)
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let (mut adapter_record, _core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());
        A::start(*state.pc, state.memory, &mut adapter_record);
        self.adapter.write(
            state.memory,
            instruction,
            self.expected_def_vks_commit,
            &mut adapter_record,
        );
        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}

impl<F, A> TraceFiller<F> for DeferralSetupCoreFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least A::WIDTH +
        // DeferralSetupCoreCols::width() elements
        let (adapter_row, core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let core_row: &mut DeferralSetupCoreCols<F> = core_row.borrow_mut();
        core_row.is_valid = F::ONE;
    }
}
