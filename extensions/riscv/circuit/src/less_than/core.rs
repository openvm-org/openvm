use std::{array, borrow::Borrow};

use openvm_circuit::arch::*;
use openvm_circuit_primitives::{
    utils::not,
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    ColumnsAir, StructReflection, StructReflectionHelper, U16_BITS,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_riscv_transpiler::LessThanOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing},
    BaseAirWithPublicValues,
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct LessThanCoreCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],
    pub cmp_result: T,

    /// Packs the opcode flags into one column:
    /// 0 = padding, 1 = SLTU (unsigned), 2 = SLT (signed).
    pub opcode_mode: T,

    // Most significant limb of b and c respectively as a field element, will be range
    // checked to be within [-2^(LIMB_BITS - 1), 2^(LIMB_BITS - 1)) if signed,
    // [0, 2^LIMB_BITS) if unsigned.
    pub b_msb_f: T,
    pub c_msb_f: T,

    // 1 at the most significant index i such that b[i] != c[i], otherwise 0. If such
    // an i exists, diff_val = c[i] - b[i] if c[i] > b[i] or b[i] - c[i] else.
    pub diff_marker: [T; NUM_LIMBS],
    pub diff_val: T,
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(LessThanCoreCols<u8, NUM_LIMBS, LIMB_BITS>)]
pub struct LessThanCoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub range_bus: VariableRangeCheckerBus,
    offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for LessThanCoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        LessThanCoreCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}
impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for LessThanCoreAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for LessThanCoreAir<NUM_LIMBS, LIMB_BITS>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; NUM_LIMBS]; 2]>,
    I::Writes: From<[[AB::Expr; NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &LessThanCoreCols<_, NUM_LIMBS, LIMB_BITS> = local_core.borrow();

        // opcode_mode is 0 (padding), 1 (SLTU) or 2 (SLT).
        let mode = cols.opcode_mode;
        builder.assert_zero(mode * (mode - AB::Expr::ONE) * (mode - AB::Expr::TWO));

        // mode * (3 - mode) / 2 evaluates to 0, 1, 1 at mode = 0, 1, 2.
        let half = AB::Expr::from(AB::F::TWO.inverse());
        let is_valid = mode * (AB::Expr::from_u32(3) - mode) * half.clone();
        // mode * (mode - 1) / 2 evaluates to 0, 0, 1 at mode = 0, 1, 2.
        let is_signed = mode * (mode - AB::Expr::ONE) * half;
        let is_unsigned = is_valid.clone() - is_signed.clone();

        builder.assert_bool(cols.cmp_result);

        let b = &cols.b;
        let c = &cols.c;
        let marker = &cols.diff_marker;
        let mut prefix_sum = AB::Expr::ZERO;

        let b_diff = b[NUM_LIMBS - 1] - cols.b_msb_f;
        let c_diff = c[NUM_LIMBS - 1] - cols.c_msb_f;
        builder.assert_zero(b_diff.clone() * (AB::Expr::from_u32(1 << LIMB_BITS) - b_diff));
        builder.assert_zero(c_diff.clone() * (AB::Expr::from_u32(1 << LIMB_BITS) - c_diff));

        for i in (0..NUM_LIMBS).rev() {
            let diff = (if i == NUM_LIMBS - 1 {
                cols.c_msb_f - cols.b_msb_f
            } else {
                c[i] - b[i]
            }) * (AB::Expr::from_u8(2) * cols.cmp_result - AB::Expr::ONE);
            prefix_sum += marker[i].into();
            builder.assert_bool(marker[i]);
            builder.assert_zero(not::<AB::Expr>(prefix_sum.clone()) * diff.clone());
            builder.when(marker[i]).assert_eq(cols.diff_val, diff);
        }
        // - If x != y, then prefix_sum = 1 so marker[i] must be 1 iff i is the first index where
        //   diff != 0. Constrains that diff == diff_val where diff_val is non-zero.
        // - If x == y, then prefix_sum = 0 and cmp_result = 0. Here, prefix_sum cannot be 1 because
        //   all diff are zero, making diff == diff_val fails.

        builder.assert_bool(prefix_sum.clone());
        builder
            .when(not::<AB::Expr>(prefix_sum.clone()))
            .assert_zero(cols.cmp_result);

        // Check if b_msb_f and c_msb_f are in
        // [-2^(LIMB_BITS - 1), 2^(LIMB_BITS - 1)) if signed, [0, 2^LIMB_BITS) if unsigned.
        let sign_shift = AB::Expr::from_u32(1 << (LIMB_BITS - 1)) * is_signed.clone();
        self.range_bus
            .range_check(cols.b_msb_f + sign_shift.clone(), LIMB_BITS)
            .eval(builder, is_valid.clone());
        self.range_bus
            .range_check(cols.c_msb_f + sign_shift, LIMB_BITS)
            .eval(builder, is_valid.clone());

        // Range check to ensure diff_val is non-zero.
        self.range_bus
            .range_check(cols.diff_val - AB::Expr::ONE, LIMB_BITS)
            .eval(builder, prefix_sum);

        let expected_opcode = is_signed * AB::Expr::from_u8(LessThanOpcode::SLT as u8)
            + is_unsigned * AB::Expr::from_u8(LessThanOpcode::SLTU as u8)
            + AB::Expr::from_usize(self.offset);
        let mut a: [AB::Expr; NUM_LIMBS] = array::from_fn(|_| AB::Expr::ZERO);
        a[0] = cols.cmp_result.into();

        AdapterAirContext {
            to_pc: None,
            reads: [cols.b.map(Into::into), cols.c.map(Into::into)].into(),
            writes: [a].into(),
            instruction: MinimalInstruction {
                is_valid,
                opcode: expected_opcode,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        self.offset
    }
}

#[derive(Clone, Copy, derive_new::new)]
pub struct LessThanExecutor<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub offset: usize,
}

#[derive(Clone, derive_new::new)]
pub struct LessThanFiller {
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

// Returns (cmp_result, diff_idx, x_sign, y_sign)
#[inline(always)]
pub(crate) fn run_less_than<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    is_slt: bool,
    x: &[u16; NUM_LIMBS],
    y: &[u16; NUM_LIMBS],
) -> (bool, usize, bool, bool) {
    debug_assert!((1..=U16_BITS).contains(&LIMB_BITS));
    let x_sign = (x[NUM_LIMBS - 1] >> (LIMB_BITS - 1) == 1) && is_slt;
    let y_sign = (y[NUM_LIMBS - 1] >> (LIMB_BITS - 1) == 1) && is_slt;
    for i in (0..NUM_LIMBS).rev() {
        if x[i] != y[i] {
            return ((x[i] < y[i]) ^ x_sign ^ y_sign, i, x_sign, y_sign);
        }
    }
    (false, NUM_LIMBS, x_sign, y_sign)
}
