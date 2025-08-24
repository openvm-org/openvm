use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::*,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::utils::not;
use openvm_circuit_primitives_derive::{AlignedBorrow, AlignedBytesBorrow};
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

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct BranchEqualCoreRecord<const NUM_LIMBS: usize> {
    pub a: [u8; NUM_LIMBS],
    pub b: [u8; NUM_LIMBS],
    pub imm: u32,
    pub local_opcode: u8,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct BranchEqualExecutor<A, const NUM_LIMBS: usize> {
    adapter: A,
    pub offset: usize,
    pub pc_step: u32,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct BranchEqualFiller<A, const NUM_LIMBS: usize> {
    adapter: A,
    pub offset: usize,
    pub pc_step: u32,
}

impl<F, A, RA, const NUM_LIMBS: usize> PreflightExecutor<F, RA>
    for BranchEqualExecutor<A, NUM_LIMBS>
where
    F: PrimeField32,
    A: 'static + AdapterTraceExecutor<F, ReadData: Into<[[u8; NUM_LIMBS]; 2]>, WriteData = ()>,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (
            A::RecordMut<'buf>,
            &'buf mut BranchEqualCoreRecord<NUM_LIMBS>,
        ),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", BranchEqualOpcode::from_usize(opcode - self.offset))
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let &Instruction { opcode, c: imm, .. } = instruction;

        let branch_eq_opcode = BranchEqualOpcode::from_usize(opcode.local_opcode_idx(self.offset));

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        let [rs1, rs2] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        core_record.a = rs1;
        core_record.b = rs2;
        core_record.imm = imm.as_canonical_u32();
        core_record.local_opcode = branch_eq_opcode as u8;

        if fast_run_eq(branch_eq_opcode, &rs1, &rs2) {
            *state.pc = (F::from_canonical_u32(*state.pc) + imm).as_canonical_u32();
        } else {
            *state.pc = state.pc.wrapping_add(self.pc_step);
        }

        Ok(())
    }
}

impl<F, A, const NUM_LIMBS: usize> TraceFiller<F> for BranchEqualFiller<A, NUM_LIMBS>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least A::WIDTH +
        // BranchEqualCoreCols::width() elements
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        // SAFETY: core_row contains a valid BranchEqualCoreRecord written by the executor
        // during trace generation
        let record: &BranchEqualCoreRecord<NUM_LIMBS> =
            unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut BranchEqualCoreCols<F, NUM_LIMBS> = core_row.borrow_mut();

        let (cmp_result, diff_idx, diff_inv_val) = run_eq::<F, NUM_LIMBS>(
            record.local_opcode == BranchEqualOpcode::BEQ as u8,
            &record.a,
            &record.b,
        );
        core_row.diff_inv_marker = [F::ZERO; NUM_LIMBS];
        core_row.diff_inv_marker[diff_idx] = diff_inv_val;

        core_row.opcode_bne_flag =
            F::from_bool(record.local_opcode == BranchEqualOpcode::BNE as u8);
        core_row.opcode_beq_flag =
            F::from_bool(record.local_opcode == BranchEqualOpcode::BEQ as u8);

        core_row.imm = F::from_canonical_u32(record.imm);
        core_row.cmp_result = F::from_bool(cmp_result);

        core_row.b = record.b.map(F::from_canonical_u8);
        core_row.a = record.a.map(F::from_canonical_u8);
    }
}

// Returns (cmp_result, diff_idx, x[diff_idx] - y[diff_idx])
#[inline(always)]
pub(super) fn fast_run_eq<const NUM_LIMBS: usize>(
    local_opcode: BranchEqualOpcode,
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> bool {
    match local_opcode {
        BranchEqualOpcode::BEQ => x == y,
        BranchEqualOpcode::BNE => x != y,
    }
}

// Returns (cmp_result, diff_idx, x[diff_idx] - y[diff_idx])
#[inline(always)]
pub(super) fn run_eq<F, const NUM_LIMBS: usize>(
    is_beq: bool,
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> (bool, usize, F)
where
    F: PrimeField32,
{
    for i in 0..NUM_LIMBS {
        if x[i] != y[i] {
            return (
                !is_beq,
                i,
                (F::from_canonical_u8(x[i]) - F::from_canonical_u8(y[i])).inverse(),
            );
        }
    }
    (is_beq, 0, F::ZERO)
}
