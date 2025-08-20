use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::*,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_native_compiler::NativeBranchEqualOpcode;
use openvm_rv32im_circuit::BranchEqualCoreCols;
use openvm_rv32im_transpiler::BranchEqualOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct NativeBranchEqualCoreRecord<F> {
    pub a: F,
    pub b: F,
    pub imm: F,
    pub is_beq: bool,
}

#[derive(derive_new::new, Clone, Copy)]
pub struct NativeBranchEqualExecutor<A> {
    adapter: A,
    pub offset: usize,
    pub pc_step: u32,
}

#[derive(derive_new::new)]
pub struct NativeBranchEqualFiller<A> {
    adapter: A,
}

impl<F, A, RA> PreflightExecutor<F, RA> for NativeBranchEqualExecutor<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceExecutor<F, ReadData: Into<[F; 2]>, WriteData = ()>,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut NativeBranchEqualCoreRecord<F>),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            NativeBranchEqualOpcode::from_usize(opcode - self.offset)
        )
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let &Instruction { opcode, c: imm, .. } = instruction;
        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        [core_record.a, core_record.b] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        let cmp_result = core_record.a == core_record.b;

        core_record.imm = imm;
        core_record.is_beq =
            opcode.local_opcode_idx(self.offset) == BranchEqualOpcode::BEQ as usize;

        if cmp_result == core_record.is_beq {
            *state.pc = (F::from_canonical_u32(*state.pc) + imm).as_canonical_u32();
        } else {
            *state.pc = state.pc.wrapping_add(self.pc_step);
        }

        Ok(())
    }
}

impl<F, A> TraceFiller<F> for NativeBranchEqualFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least A::WIDTH +
        // BranchEqualCoreCols::width() elements
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        // SAFETY: core_row contains a valid NativeBranchEqualCoreRecord written by the executor
        // during trace generation
        let record: &NativeBranchEqualCoreRecord<F> =
            unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut BranchEqualCoreCols<F, 1> = core_row.borrow_mut();
        let (cmp_result, diff_inv_val) = run_eq(record.is_beq, record.a, record.b);

        // Writing in reverse order to avoid overwriting the `record`
        core_row.diff_inv_marker[0] = diff_inv_val;

        core_row.opcode_bne_flag = F::from_bool(!record.is_beq);
        core_row.opcode_beq_flag = F::from_bool(record.is_beq);

        core_row.imm = record.imm;
        core_row.cmp_result = F::from_bool(cmp_result);

        core_row.b = [record.b];
        core_row.a = [record.a];
    }
}

// Returns (cmp_result, diff_idx, x[diff_idx] - y[diff_idx])
#[inline(always)]
pub(super) fn run_eq<F>(is_beq: bool, x: F, y: F) -> (bool, F)
where
    F: PrimeField32,
{
    if x != y {
        return (!is_beq, (x - y).inverse());
    }
    (is_beq, F::ZERO)
}
