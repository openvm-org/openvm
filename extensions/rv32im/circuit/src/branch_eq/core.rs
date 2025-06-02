use std::{
    array,
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit::{
    arch::{
        execution_mode::{metered::MeteredCtx, E1E2ExecutionCtx},
        AdapterAirContext, AdapterExecutorE1, AdapterTraceStep, ImmInstruction, Result,
        StepExecutorE1, TraceStep, VmAdapterInterface, VmCoreAir, VmStateMut,
    },
    system::memory::{
        online::{GuestMemory, TracingMemory},
        MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::utils::not;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_rv32im_transpiler::BranchEqualOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};
use strum::IntoEnumIterator;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct BranchEqualCoreCols<T, const NUM_LIMBS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],

    // Boolean result of a op b. Should branch if and only if cmp_result = 1.
    pub cmp_result: T,
    pub imm: T,

    pub opcode_beq_flag: T,
    pub opcode_bne_flag: T,

    pub diff_inv_marker: [T; NUM_LIMBS],
}

#[derive(Copy, Clone, Debug, derive_new::new)]
pub struct BranchEqualCoreAir<const NUM_LIMBS: usize> {
    offset: usize,
    pc_step: u32,
}

impl<F: Field, const NUM_LIMBS: usize> BaseAir<F> for BranchEqualCoreAir<NUM_LIMBS> {
    fn width(&self) -> usize {
        BranchEqualCoreCols::<F, NUM_LIMBS>::width()
    }
}
impl<F: Field, const NUM_LIMBS: usize> BaseAirWithPublicValues<F>
    for BranchEqualCoreAir<NUM_LIMBS>
{
}

impl<AB, I, const NUM_LIMBS: usize> VmCoreAir<AB, I> for BranchEqualCoreAir<NUM_LIMBS>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; NUM_LIMBS]; 2]>,
    I::Writes: Default,
    I::ProcessedInstruction: From<ImmInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &BranchEqualCoreCols<_, NUM_LIMBS> = local.borrow();
        let flags = [cols.opcode_beq_flag, cols.opcode_bne_flag];

        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());
        builder.assert_bool(cols.cmp_result);

        let a = &cols.a;
        let b = &cols.b;
        let inv_marker = &cols.diff_inv_marker;

        // 1 if cmp_result indicates a and b are equal, 0 otherwise
        let cmp_eq =
            cols.cmp_result * cols.opcode_beq_flag + not(cols.cmp_result) * cols.opcode_bne_flag;
        let mut sum = cmp_eq.clone();

        // For BEQ, inv_marker is used to check equality of a and b:
        // - If a == b, all inv_marker values must be 0 (sum = 0)
        // - If a != b, inv_marker contains 0s for all positions except ONE position i where a[i] !=
        //   b[i]
        // - At this position, inv_marker[i] contains the multiplicative inverse of (a[i] - b[i])
        // - This ensures inv_marker[i] * (a[i] - b[i]) = 1, making the sum = 1
        // Note: There might be multiple valid inv_marker if a != b.
        // But as long as the trace can provide at least one, that’s sufficient to prove a != b.
        //
        // Note:
        // - If cmp_eq == 0, then it is impossible to have sum != 0 if a == b.
        // - If cmp_eq == 1, then it is impossible for a[i] - b[i] == 0 to pass for all i if a != b.
        for i in 0..NUM_LIMBS {
            sum += (a[i] - b[i]) * inv_marker[i];
            builder.assert_zero(cmp_eq.clone() * (a[i] - b[i]));
        }
        builder.when(is_valid.clone()).assert_one(sum);

        let expected_opcode = flags
            .iter()
            .zip(BranchEqualOpcode::iter())
            .fold(AB::Expr::ZERO, |acc, (flag, opcode)| {
                acc + (*flag).into() * AB::Expr::from_canonical_u8(opcode as u8)
            })
            + AB::Expr::from_canonical_usize(self.offset);

        let to_pc = from_pc
            + cols.cmp_result * cols.imm
            + not(cols.cmp_result) * AB::Expr::from_canonical_u32(self.pc_step);

        AdapterAirContext {
            to_pc: Some(to_pc),
            reads: [cols.a.map(Into::into), cols.b.map(Into::into)].into(),
            writes: Default::default(),
            instruction: ImmInstruction {
                is_valid,
                opcode: expected_opcode,
                immediate: cols.imm.into(),
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        self.offset
    }
}

pub struct BranchEqualStep<A, const NUM_LIMBS: usize> {
    adapter: A,
    pub offset: usize,
    pub pc_step: u32,
}

impl<A, const NUM_LIMBS: usize> BranchEqualStep<A, NUM_LIMBS> {
    pub fn new(adapter: A, offset: usize, pc_step: u32) -> Self {
        Self {
            adapter,
            offset,
            pc_step,
        }
    }
}

impl<F, CTX, A, const NUM_LIMBS: usize> TraceStep<F, CTX> for BranchEqualStep<A, NUM_LIMBS>
where
    F: PrimeField32,
    A: 'static
        + for<'a> AdapterTraceStep<
            F,
            CTX,
            ReadData: Into<[[u8; NUM_LIMBS]; 2]>,
            WriteData = (),
            TraceContext<'a> = (),
        >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", BranchEqualOpcode::from_usize(opcode - self.offset))
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

        let branch_eq_opcode = BranchEqualOpcode::from_usize(opcode.local_opcode_idx(self.offset));

        let row_slice = &mut trace[*trace_offset..*trace_offset + width];
        let (adapter_row, core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };

        A::start(*state.pc, state.memory, adapter_row);

        let [rs1, rs2] = self
            .adapter
            .read(state.memory, instruction, adapter_row)
            .into();

        let (cmp_result, diff_idx, diff_inv_val) = run_eq(branch_eq_opcode, &rs1, &rs2);

        let core_row: &mut BranchEqualCoreCols<_, NUM_LIMBS> = core_row.borrow_mut();
        core_row.a = rs1.map(F::from_canonical_u8);
        core_row.b = rs2.map(F::from_canonical_u8);
        core_row.cmp_result = F::from_bool(cmp_result);
        core_row.imm = imm;
        core_row.opcode_beq_flag = F::from_bool(branch_eq_opcode == BranchEqualOpcode::BEQ);
        core_row.opcode_bne_flag = F::from_bool(branch_eq_opcode == BranchEqualOpcode::BNE);
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

impl<F, A, const NUM_LIMBS: usize> StepExecutorE1<F> for BranchEqualStep<A, NUM_LIMBS>
where
    F: PrimeField32,
    A: 'static + for<'a> AdapterExecutorE1<F, ReadData: Into<[[u8; NUM_LIMBS]; 2]>, WriteData = ()>,
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

        let branch_eq_opcode = BranchEqualOpcode::from_usize(opcode.local_opcode_idx(self.offset));

        let [rs1, rs2] = self.adapter.read(state, instruction).into();

        // TODO(ayush): probably don't need the other values
        let (cmp_result, _, _) = run_eq::<F, NUM_LIMBS>(branch_eq_opcode, &rs1, &rs2);

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
pub(super) fn run_eq<F, const NUM_LIMBS: usize>(
    local_opcode: BranchEqualOpcode,
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> (bool, usize, F)
where
    F: PrimeField32,
{
    for i in 0..NUM_LIMBS {
        if x[i] != y[i] {
            return (
                local_opcode == BranchEqualOpcode::BNE,
                i,
                (F::from_canonical_u8(x[i]) - F::from_canonical_u8(y[i])).inverse(),
            );
        }
    }
    (local_opcode == BranchEqualOpcode::BEQ, 0, F::ZERO)
}
