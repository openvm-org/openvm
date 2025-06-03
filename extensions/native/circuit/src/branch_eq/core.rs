use std::{array, borrow::BorrowMut};

use openvm_circuit::{
    arch::{
        execution_mode::{metered::MeteredCtx, E1E2ExecutionCtx},
        AdapterExecutorE1, AdapterTraceStep, Result, StepExecutorE1, TraceStep, VmStateMut,
    },
    system::memory::{
        online::{GuestMemory, TracingMemory},
        MemoryAuxColsFactory,
    },
};
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_native_compiler::NativeBranchEqualOpcode;
use openvm_rv32im_circuit::BranchEqualCoreCols;
use openvm_rv32im_transpiler::BranchEqualOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

pub struct NativeBranchEqualStep<A> {
    adapter: A,
    pub offset: usize,
    pub pc_step: u32,
}

impl<A> NativeBranchEqualStep<A> {
    pub fn new(adapter: A, offset: usize, pc_step: u32) -> Self {
        Self {
            adapter,
            offset,
            pc_step,
        }
    }
}

impl<F, CTX, A> TraceStep<F, CTX> for NativeBranchEqualStep<A>
where
    F: PrimeField32,
    A: 'static
        + for<'a> AdapterTraceStep<
            F,
            CTX,
            ReadData: Into<[F; 2]>,
            WriteData = (),
            TraceContext<'a> = (),
        >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            NativeBranchEqualOpcode::from_usize(opcode - self.offset)
        )
    }

    fn execute(
        &mut self,
        state: VmStateMut<TracingMemory<F>, CTX>,
        instruction: &Instruction<F>,
        trace: &mut [F],
        trace_offset: &mut usize,
        width: usize,
    ) -> Result<()> {
        let &Instruction { opcode, c: imm, .. } = instruction;

        let branch_eq_opcode =
            NativeBranchEqualOpcode::from_usize(opcode.local_opcode_idx(self.offset));

        let row_slice = &mut trace[*trace_offset..*trace_offset + width];
        let (adapter_row, core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };

        A::start(*state.pc, state.memory, adapter_row);

        let [rs1, rs2] = self
            .adapter
            .read(state.memory, instruction, adapter_row)
            .into();

        let (cmp_result, diff_idx, diff_inv_val) = run_eq(branch_eq_opcode, rs1, rs2);

        let core_row: &mut BranchEqualCoreCols<_, 1> = core_row.borrow_mut();
        core_row.a = [rs1];
        core_row.b = [rs2];
        core_row.cmp_result = F::from_bool(cmp_result);
        core_row.imm = imm;
        core_row.opcode_beq_flag = F::from_bool(branch_eq_opcode.0 == BranchEqualOpcode::BEQ);
        core_row.opcode_bne_flag = F::from_bool(branch_eq_opcode.0 == BranchEqualOpcode::BNE);
        core_row.diff_inv_marker =
            array::from_fn(|i| if i == diff_idx { diff_inv_val } else { F::ZERO });

        if cmp_result {
            *state.pc = (F::from_canonical_u32(*state.pc) + imm).as_canonical_u32();
        } else {
            *state.pc = state.pc.wrapping_add(self.pc_step);
        }

        *trace_offset += width;

        Ok(())
    }

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, _core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };

        self.adapter.fill_trace_row(mem_helper, (), adapter_row);
    }
}

impl<F, A> StepExecutorE1<F> for NativeBranchEqualStep<A>
where
    F: PrimeField32,
    A: 'static + for<'a> AdapterExecutorE1<F, ReadData: Into<[F; 2]>, WriteData = ()>,
{
    fn execute_e1<Ctx>(
        &self,
        state: &mut VmStateMut<GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<()>
    where
        Ctx: E1E2ExecutionCtx,
    {
        let &Instruction { opcode, c: imm, .. } = instruction;

        let branch_eq_opcode =
            NativeBranchEqualOpcode::from_usize(opcode.local_opcode_idx(self.offset));

        let [rs1, rs2] = self.adapter.read(state, instruction).into();

        // TODO(ayush): probably don't need the other values
        let (cmp_result, _, _) = run_eq::<F>(branch_eq_opcode, rs1, rs2);

        if cmp_result {
            // TODO(ayush): verify this is fine
            // state.pc = state.pc.wrapping_add(imm.as_canonical_u32());
            *state.pc = (F::from_canonical_u32(*state.pc) + imm).as_canonical_u32();
        } else {
            *state.pc = state.pc.wrapping_add(self.pc_step);
        }

        Ok(())
    }

    fn execute_metered(
        &self,
        state: &mut VmStateMut<GuestMemory, MeteredCtx>,
        instruction: &Instruction<F>,
        chip_index: usize,
    ) -> Result<()> {
        self.execute_e1(state, instruction)?;
        state.ctx.trace_heights[chip_index] += 1;

        Ok(())
    }
}

// Returns (cmp_result, diff_idx, x[diff_idx] - y[diff_idx])
#[inline(always)]
pub(super) fn run_eq<F>(local_opcode: NativeBranchEqualOpcode, x: F, y: F) -> (bool, usize, F)
where
    F: PrimeField32,
{
    if x != y {
        return (
            local_opcode.0 == BranchEqualOpcode::BNE,
            0,
            (x - y).inverse(),
        );
    }
    (local_opcode.0 == BranchEqualOpcode::BEQ, 0, F::ZERO)
}
