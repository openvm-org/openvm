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

    /// Packs the opcode flags into one column:
    /// 0 = padding, 1 = SLTIU (unsigned), 2 = SLTI (signed).
    pub opcode_mode: T,

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
        // opcode_mode is 0 (padding), 1 (SLTIU) or 2 (SLTI).
        let mode = cols.opcode_mode;
        builder.assert_zero(mode * (mode - AB::Expr::ONE) * (mode - AB::Expr::TWO));

        // mode * (3 - mode) / 2 evaluates to 0, 1, 1 at mode = 0, 1, 2
        let is_valid = mode * (AB::Expr::from_u32(3) - mode) * AB::Expr::from(AB::F::TWO.inverse());

        // `mode - 1` is 0 for SLTIU and 1 for SLTI, giving the signed selector at degree 1. The
        // exact form `mode * (mode - 1) / 2` is degree 2, which would push
        // `c_msb_f` to degree 3 and the two constraints reading it to degree 4.
        //
        // On a mode=0 row, `is_signed` and `is_unsigned` are -1 and 2 rather than 0. `sign_shift`
        // and `expected_opcode` are unaffected because their interactions have count
        // `is_valid`, which is exactly 0 there. `c_msb_f` is the one ungated consumer, so
        // `imm_sign` is pinned to zero on invalid rows below, which forces `c_msb_f` to
        // zero too.
        let is_signed = mode - AB::Expr::ONE;
        let is_unsigned = AB::Expr::ONE - is_signed.clone();

        builder.assert_bool(cols.cmp_result);
        builder.assert_bool(cols.imm_sign);
        // An invalid row must have imm_sign = 0.
        builder
            .when(not::<AB::Expr>(is_valid.clone()))
            .assert_zero(cols.imm_sign);

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
        // For padding rows, `c_msb_f=0` due to imm_sign being 0.
        let c_msb_f: AB::Expr = cols.imm_sign
            * (AB::Expr::from_u32((1 << LIMB_BITS) - 1)
                - is_signed.clone() * AB::Expr::from_u32(1 << LIMB_BITS));

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

        // For padding rows, `sign_shift` is unaffected by `is_signed` being -2.
        // This is due to is_valid gating.
        let sign_shift = AB::Expr::from_u32(1 << (LIMB_BITS - 1)) * is_signed.clone();
        self.range_bus
            .range_check(cols.b_msb_f + sign_shift.clone(), LIMB_BITS)
            .eval(builder, is_valid.clone());

        self.range_bus
            .range_check(cols.diff_val - AB::Expr::ONE, LIMB_BITS)
            .eval(builder, prefix_sum);

        // For padding rows, `expected_opcode` is gated by `is_valid`. Hence, its content can be
        // anything.
        let expected_opcode = is_signed * AB::Expr::from_u8(LessThanImmOpcode::SLTI as u8)
            + is_unsigned * AB::Expr::from_u8(LessThanImmOpcode::SLTIU as u8)
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
pub struct LessThanImmFiller {
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
