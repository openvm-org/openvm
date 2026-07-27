use std::{array, borrow::Borrow};

use openvm_circuit::arch::*;
use openvm_circuit_primitives::{
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing},
    BaseAirWithPublicValues,
};

use crate::adapters::U16_BITS;

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct AddICoreCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    /// Result limbs.
    pub rd: [T; NUM_LIMBS],
    /// Source operand limbs.
    pub rs1: [T; NUM_LIMBS],
    /// Low 11 bits (`imm[10:0]`) of the signed 12-bit immediate.
    pub imm_low11: T,
    /// Sign bit (`imm[11]`), used to sign-extend the immediate across all limbs.
    pub imm_sign: T,
    /// Whether this row contains an instruction.
    pub is_valid: T,
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(AddICoreCols<u8, NUM_LIMBS, LIMB_BITS>)]
pub struct AddICoreAir<
    const NUM_LIMBS: usize,
    const LIMB_BITS: usize,
    const RANGE_CHECK_TOP_LIMB: bool,
> {
    pub range_bus: VariableRangeCheckerBus,
    pub offset: usize,
    pub local_opcode: usize,
}

impl<
        F: Field,
        const NUM_LIMBS: usize,
        const LIMB_BITS: usize,
        const RANGE_CHECK_TOP_LIMB: bool,
    > BaseAir<F> for AddICoreAir<NUM_LIMBS, LIMB_BITS, RANGE_CHECK_TOP_LIMB>
{
    fn width(&self) -> usize {
        AddICoreCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}
impl<
        F: Field,
        const NUM_LIMBS: usize,
        const LIMB_BITS: usize,
        const RANGE_CHECK_TOP_LIMB: bool,
    > BaseAirWithPublicValues<F> for AddICoreAir<NUM_LIMBS, LIMB_BITS, RANGE_CHECK_TOP_LIMB>
{
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize, const RANGE_CHECK_TOP_LIMB: bool>
    VmCoreAir<AB, I> for AddICoreAir<NUM_LIMBS, LIMB_BITS, RANGE_CHECK_TOP_LIMB>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; NUM_LIMBS]; 1]>,
    I::Writes: From<[[AB::Expr; NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<ImmInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        assert!(NUM_LIMBS > 0 && (12..=U16_BITS).contains(&LIMB_BITS));

        let cols: &AddICoreCols<_, NUM_LIMBS, LIMB_BITS> = local_core.borrow();

        builder.assert_bool(cols.is_valid);
        builder.assert_bool(cols.imm_sign);

        self.range_bus
            .range_check(cols.imm_low11, 11)
            .eval(builder, cols.is_valid.into());

        let limb_base = 1usize << LIMB_BITS;
        let limb_mask = limb_base - 1;
        let imm_sign: AB::Expr = cols.imm_sign.into();
        let sign_limb: AB::Expr = imm_sign.clone() * AB::Expr::from_usize(limb_mask);
        let imm0: AB::Expr =
            cols.imm_low11 + imm_sign.clone() * AB::Expr::from_usize(limb_base - (1 << 11));

        let carry_divide = AB::F::from_usize(limb_base).inverse();
        let mut carry: [AB::Expr; NUM_LIMBS] = array::from_fn(|_| AB::Expr::ZERO);

        carry[0] = AB::Expr::from(carry_divide) * (cols.rs1[0] + imm0 - cols.rd[0]);
        builder.when(cols.is_valid).assert_bool(carry[0].clone());

        for i in 1..NUM_LIMBS {
            carry[i] = AB::Expr::from(carry_divide)
                * (cols.rs1[i] + sign_limb.clone() - cols.rd[i] + carry[i - 1].clone());
            builder.when(cols.is_valid).assert_bool(carry[i].clone());
        }

        let range_limb_count = NUM_LIMBS - usize::from(!RANGE_CHECK_TOP_LIMB);
        for &rd_limb in &cols.rd[..range_limb_count] {
            self.range_bus
                .range_check(rd_limb, LIMB_BITS)
                .eval(builder, cols.is_valid.into());
        }

        // 24-bit encoding matching i12_to_u24 in the transpiler.
        let instr_c: AB::Expr = cols.imm_low11 + imm_sign * AB::Expr::from_u32(0xFFF800);

        let expected_opcode =
            VmCoreAir::<AB, I>::expr_to_global_expr(self, AB::Expr::from_usize(self.local_opcode));

        AdapterAirContext {
            to_pc: None,
            reads: [cols.rs1.map(Into::into)].into(),
            writes: [cols.rd.map(Into::into)].into(),
            instruction: ImmInstruction {
                is_valid: cols.is_valid.into(),
                opcode: expected_opcode,
                immediate: instr_c,
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
pub struct AddICoreRecord<const NUM_LIMBS: usize> {
    pub rs1: [u16; NUM_LIMBS],
    pub imm_low11: u16,
    pub imm_sign: u16,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct AddIExecutor<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub offset: usize,
    pub local_opcode: usize,
}

#[derive(derive_new::new)]
pub struct AddIFiller<
    A,
    const NUM_LIMBS: usize,
    const LIMB_BITS: usize,
    const RANGE_CHECK_TOP_LIMB: bool,
> {
    adapter: A,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

#[inline(always)]
pub(crate) fn run_addi<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    rs1: &[u16; NUM_LIMBS],
    imm_low11: u16,
    imm_sign: u16,
) -> [u16; NUM_LIMBS] {
    debug_assert!(NUM_LIMBS > 0 && (12..=U16_BITS).contains(&LIMB_BITS));

    let mut z = [0u16; NUM_LIMBS];
    let limb_base = 1u32 << LIMB_BITS;
    let limb_mask = limb_base - 1;

    let mut overflow = rs1[0] as u32 + imm_low11 as u32 + imm_sign as u32 * (limb_base - (1 << 11));
    let mut carry = overflow >> LIMB_BITS;
    z[0] = (overflow & limb_mask) as u16;

    let sign_limb = imm_sign as u32 * limb_mask;
    for i in 1..NUM_LIMBS {
        overflow = rs1[i] as u32 + sign_limb + carry;
        carry = overflow >> LIMB_BITS;
        z[i] = (overflow & limb_mask) as u16;
    }
    z
}
