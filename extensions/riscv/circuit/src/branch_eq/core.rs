use std::borrow::Borrow;

use openvm_circuit::arch::*;
use openvm_circuit_primitives::{utils::not, ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::program::DEFAULT_PC_STEP;
use openvm_riscv_transpiler::BranchEqualOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};
use strum::IntoEnumIterator;

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
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

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(BranchEqualCoreCols<u8, NUM_LIMBS>)]
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

        // The adapter reads u16 memory cells, so the memory bus enforces the limb bounds.
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
        // But as long as the trace can provide at least one, that's sufficient to prove a != b.
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
                acc + (*flag).into() * AB::Expr::from_u8(opcode as u8)
            })
            + AB::Expr::from_usize(self.offset);

        // `imm` is a byte offset (a multiple of DEFAULT_PC_STEP, possibly negative as a field
        // element) and `pc_step` is a byte step; pc values on the buses are pc indices, so the
        // byte delta is scaled down by DEFAULT_PC_STEP.
        let pc_step_inv = AB::F::from_u32(DEFAULT_PC_STEP).inverse();
        let to_pc = from_pc
            + (cols.cmp_result * cols.imm
                + not(cols.cmp_result) * AB::Expr::from_u32(self.pc_step))
                * pc_step_inv;

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

#[derive(Clone, Copy, derive_new::new)]
pub struct BranchEqualCoreExecutor<const NUM_LIMBS: usize> {
    pub offset: usize,
    pub pc_step: u32,
}

#[derive(Clone, derive_new::new)]
pub struct BranchEqualFiller {
    pub pc_step: u32,
}

// Returns (cmp_result, diff_idx, x[diff_idx] - y[diff_idx])
#[inline(always)]
pub(super) fn fast_run_eq<const NUM_LIMBS: usize>(
    local_opcode: BranchEqualOpcode,
    x: &[u16; NUM_LIMBS],
    y: &[u16; NUM_LIMBS],
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
    x: &[u16; NUM_LIMBS],
    y: &[u16; NUM_LIMBS],
) -> (bool, usize, F)
where
    F: PrimeField32,
{
    for i in 0..NUM_LIMBS {
        if x[i] != y[i] {
            return (
                !is_beq,
                i,
                (F::from_u16(x[i]) - F::from_u16(y[i])).inverse(),
            );
        }
    }
    (is_beq, 0, F::ZERO)
}
