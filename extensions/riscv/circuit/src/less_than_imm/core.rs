use std::{array, borrow::Borrow};

use openvm_circuit::arch::*;
use openvm_circuit_primitives::{
    utils::not,
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_riscv_transpiler::LessThanImmOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing},
    BaseAirWithPublicValues,
};
use strum::IntoEnumIterator;

/// Core columns for comparisons with a signed 12-bit immediate.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct LessThanImmCoreCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub b: [T; NUM_LIMBS],
    /// Bits `[10:0]` of the 12-bit signed immediate.
    pub imm_low11: T,
    /// Sign bit (bit 11) of the immediate.
    pub imm_sign: T,
    pub cmp_result: T,

    pub opcode_slt_flag: T,
    pub opcode_sltu_flag: T,

    pub b_msb_f: T,

    pub diff_marker: [T; NUM_LIMBS],
    pub diff_val: T,
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(LessThanImmCoreCols<u8, NUM_LIMBS, LIMB_BITS>)]
pub struct LessThanImmCoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub range_bus: VariableRangeCheckerBus,
    offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for LessThanImmCoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        LessThanImmCoreCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}
impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for LessThanImmCoreAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for LessThanImmCoreAir<NUM_LIMBS, LIMB_BITS>
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
        let cols: &LessThanImmCoreCols<_, NUM_LIMBS, LIMB_BITS> = local_core.borrow();
        let flags = [cols.opcode_slt_flag, cols.opcode_sltu_flag];

        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());
        builder.assert_bool(cols.cmp_result);
        builder.assert_bool(cols.imm_sign);

        // Range check the low 11 bits of the immediate so the (imm_low11, imm_sign)
        // decomposition of the 24-bit operand is unique.
        self.range_bus
            .range_check(cols.imm_low11, 11)
            .eval(builder, is_valid.clone());

        // Sign-extended u16 limbs of the immediate, as expressions.
        let sign_u16 = cols.imm_sign * AB::Expr::from_u32(u16::MAX as u32);
        let c: [AB::Expr; NUM_LIMBS] = array::from_fn(|i| {
            if i == 0 {
                cols.imm_low11 + cols.imm_sign * AB::Expr::from_u32(0xF800)
            } else {
                sign_u16.clone()
            }
        });

        let b = &cols.b;
        let marker = &cols.diff_marker;
        let mut prefix_sum = AB::Expr::ZERO;

        let b_diff = b[NUM_LIMBS - 1] - cols.b_msb_f;
        builder.assert_zero(b_diff.clone() * (AB::Expr::from_u32(1 << LIMB_BITS) - b_diff));

        // Field representation of the immediate's top limb for signed or unsigned comparison.
        let c_msb_f: AB::Expr = cols.imm_sign
            * (AB::Expr::from_u32((1 << LIMB_BITS) - 1)
                - cols.opcode_slt_flag * AB::Expr::from_u32(1 << LIMB_BITS));

        // Maps cmp_result to -1 or 1, so cmp_sign^2 = 1.
        let cmp_sign = AB::Expr::from_u8(2) * cols.cmp_result - AB::Expr::ONE;

        // Multiplying diff_val by cmp_sign keeps the constraint degree at 3 because c_msb_f has
        // degree 2, and is equivalent to diff_val = cmp_sign * raw_diff.
        for i in (0..NUM_LIMBS).rev() {
            let raw_diff = if i == NUM_LIMBS - 1 {
                c_msb_f.clone() - cols.b_msb_f
            } else {
                c[i].clone() - b[i]
            };
            prefix_sum += marker[i].into();
            builder.assert_bool(marker[i]);
            builder.assert_zero(not::<AB::Expr>(prefix_sum.clone()) * raw_diff.clone());
            builder
                .when(marker[i])
                .assert_eq(cmp_sign.clone() * cols.diff_val, raw_diff);
        }

        builder.assert_bool(prefix_sum.clone());
        builder
            .when(not::<AB::Expr>(prefix_sum.clone()))
            .assert_zero(cols.cmp_result);

        let sign_shift = AB::Expr::from_u32(1 << (LIMB_BITS - 1)) * cols.opcode_slt_flag;
        self.range_bus
            .range_check(cols.b_msb_f + sign_shift.clone(), LIMB_BITS)
            .eval(builder, is_valid.clone());

        self.range_bus
            .range_check(cols.diff_val - AB::Expr::ONE, LIMB_BITS)
            .eval(builder, prefix_sum);

        let expected_opcode = flags
            .iter()
            .zip(LessThanImmOpcode::iter())
            .fold(AB::Expr::ZERO, |acc, (flag, opcode)| {
                acc + (*flag).into() * AB::Expr::from_u8(opcode as u8)
            })
            + AB::Expr::from_usize(self.offset);
        let mut a: [AB::Expr; NUM_LIMBS] = array::from_fn(|_| AB::Expr::ZERO);
        a[0] = cols.cmp_result.into();

        // 24-bit encoding matching i12_to_u24 in the transpiler.
        let imm = cols.imm_low11 + cols.imm_sign * AB::Expr::from_u32(0xFFF800);

        AdapterAirContext {
            to_pc: None,
            reads: [cols.b.map(Into::into)].into(),
            writes: [a].into(),
            instruction: ImmInstruction {
                is_valid,
                opcode: expected_opcode,
                immediate: imm,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        self.offset
    }
}

#[derive(Clone, Copy, derive_new::new)]
pub struct LessThanImmExecutor<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub offset: usize,
}

#[derive(Clone, derive_new::new)]
pub struct LessThanImmFiller<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

pub(crate) fn imm_to_u16_limbs<const NUM_LIMBS: usize>(
    imm_low11: u16,
    imm_sign: u8,
) -> [u16; NUM_LIMBS] {
    let imm_sign = u16::from(imm_sign);
    let mut c = [imm_sign * 0xFFFF; NUM_LIMBS];
    c[0] = imm_low11 + imm_sign * 0xF800;
    c
}
