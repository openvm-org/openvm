use std::{borrow::Borrow, marker::PhantomData};

use openvm_circuit::arch::*;
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupBus, var_range::VariableRangeCheckerBus, ColumnsAir,
    StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_riscv_transpiler::ShiftOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing},
    BaseAirWithPublicValues,
};

use super::ShiftOp;

/// Columns shared by the [SLL](super::Sll) and [SRL](super::Srl) chips. Compared to the combined
/// shift chip these drop the three opcode flags (the opcode is fixed per chip), merge the two
/// directional `bit_multiplier`s into one, and omit the sign column (only [SRA](super::Sra) needs
/// it).
#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Clone, Copy, Debug)]
pub struct ShiftCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],

    /// `bit_multiplier = 2^bit_shift`.
    pub bit_multiplier: T,

    /// Boolean columns that are 1 exactly at the index of the bit/limb shift amount.
    pub bit_shift_marker: [T; LIMB_BITS],
    pub limb_shift_marker: [T; NUM_LIMBS],

    /// Part of each `b[i]` that gets bit shifted to the next limb.
    pub bit_shift_carry: [T; NUM_LIMBS],
}

/// Columns for the [SRA](super::Sra) chip: identical to [ShiftCols] plus the sign bit of `b`.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Clone, Copy, Debug)]
pub struct ShiftSraCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],

    /// `bit_multiplier = 2^bit_shift`.
    pub bit_multiplier: T,
    /// Sign of `b`, i.e. the MSB of `b[NUM_LIMBS - 1]`.
    pub b_sign: T,

    pub bit_shift_marker: [T; LIMB_BITS],
    pub limb_shift_marker: [T; NUM_LIMBS],

    pub bit_shift_carry: [T; NUM_LIMBS],
}

/// Shift AIR for the logical shifts [SLL](super::Sll) and [SRL](super::Srl), selected by the marker
/// type `OP`.
///
/// Note: when the shift amount from the operand is greater than the number of bits, only
/// `shift_amount % num_bits` bits are shifted. This matches the RISC-V specs for SLL/SRL/SRA.
#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(ShiftCols<u8, NUM_LIMBS, LIMB_BITS>)]
pub struct LogicalShiftCoreAir<OP, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
    pub range_bus: VariableRangeCheckerBus,
    pub offset: usize,
    #[new(default)]
    phantom: PhantomData<OP>,
}

/// Shift AIR for the arithmetic right shift [SRA](super::Sra).
#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(ShiftSraCols<u8, NUM_LIMBS, LIMB_BITS>)]
pub struct SraCoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
    pub range_bus: VariableRangeCheckerBus,
    pub offset: usize,
}

impl<F: Field, OP: Send + Sync, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for LogicalShiftCoreAir<OP, NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        ShiftCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}
impl<F: Field, OP: Send + Sync, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    BaseAirWithPublicValues<F> for LogicalShiftCoreAir<OP, NUM_LIMBS, LIMB_BITS>
{
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for SraCoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        ShiftSraCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}
impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for SraCoreAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB, I, OP, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for LogicalShiftCoreAir<OP, NUM_LIMBS, LIMB_BITS>
where
    AB: InteractionBuilder,
    OP: ShiftOp,
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
        let cols: &ShiftCols<_, NUM_LIMBS, LIMB_BITS> = local_core.borrow();

        let (is_valid, bit_shift) =
            eval_bit_shift::<AB, LIMB_BITS>(builder, &cols.bit_shift_marker, cols.bit_multiplier);
        let limb_shift =
            eval_limb_shift::<AB, NUM_LIMBS>(builder, &cols.limb_shift_marker, is_valid.clone());

        if OP::IS_LEFT {
            eval_left_limbs::<AB, NUM_LIMBS, LIMB_BITS>(
                builder,
                &cols.a,
                &cols.b,
                cols.bit_multiplier,
                &cols.bit_shift_carry,
                &cols.limb_shift_marker,
            );
        } else {
            eval_right_limbs::<AB, NUM_LIMBS, LIMB_BITS>(
                builder,
                &cols.a,
                &cols.b,
                cols.bit_multiplier,
                AB::Expr::ZERO,
                &cols.bit_shift_carry,
                &cols.limb_shift_marker,
            );
        }

        eval_ranges::<AB, NUM_LIMBS, LIMB_BITS>(
            self.bitwise_lookup_bus,
            self.range_bus,
            builder,
            &cols.a,
            &cols.b,
            &cols.c,
            &cols.bit_shift_carry,
            limb_shift,
            bit_shift,
            is_valid.clone(),
        );

        let expected_opcode =
            VmCoreAir::<AB, I>::expr_to_global_expr(self, AB::Expr::from_u8(OP::OPCODE as u8));

        AdapterAirContext {
            to_pc: None,
            reads: [cols.b.map(Into::into), cols.c.map(Into::into)].into(),
            writes: [cols.a.map(Into::into)].into(),
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

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for SraCoreAir<NUM_LIMBS, LIMB_BITS>
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
        let cols: &ShiftSraCols<_, NUM_LIMBS, LIMB_BITS> = local_core.borrow();

        let (is_valid, bit_shift) =
            eval_bit_shift::<AB, LIMB_BITS>(builder, &cols.bit_shift_marker, cols.bit_multiplier);
        let limb_shift =
            eval_limb_shift::<AB, NUM_LIMBS>(builder, &cols.limb_shift_marker, is_valid.clone());

        builder.assert_bool(cols.b_sign);
        eval_right_limbs::<AB, NUM_LIMBS, LIMB_BITS>(
            builder,
            &cols.a,
            &cols.b,
            cols.bit_multiplier,
            cols.b_sign.into(),
            &cols.bit_shift_carry,
            &cols.limb_shift_marker,
        );

        eval_ranges::<AB, NUM_LIMBS, LIMB_BITS>(
            self.bitwise_lookup_bus,
            self.range_bus,
            builder,
            &cols.a,
            &cols.b,
            &cols.c,
            &cols.bit_shift_carry,
            limb_shift,
            bit_shift,
            is_valid.clone(),
        );

        // Check `b_sign == MSB(b[NUM_LIMBS - 1])` using an XOR with the high bit mask.
        let mask = AB::F::from_u32(1 << (LIMB_BITS - 1));
        let b_sign_shifted = cols.b_sign * mask;
        self.bitwise_lookup_bus
            .send_xor(
                cols.b[NUM_LIMBS - 1],
                mask,
                cols.b[NUM_LIMBS - 1] + mask - (AB::Expr::from_u32(2) * b_sign_shifted),
            )
            .eval(builder, is_valid.clone());

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            AB::Expr::from_u8(ShiftOpcode::SRA as u8),
        );

        AdapterAirContext {
            to_pc: None,
            reads: [cols.b.map(Into::into), cols.c.map(Into::into)].into(),
            writes: [cols.a.map(Into::into)].into(),
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

/// Constrains `bit_multiplier = 1 << bit_shift` via the one-hot `bit_shift_marker`.
///
/// Returns `(is_valid, bit_shift)`. `is_valid` is the sum of all markers; it is `0` on padding rows
/// and constrained to be boolean, so on a real row (forced by the execution bus) exactly one marker
/// is set.
fn eval_bit_shift<AB: InteractionBuilder, const LIMB_BITS: usize>(
    builder: &mut AB,
    bit_shift_marker: &[AB::Var; LIMB_BITS],
    bit_multiplier: AB::Var,
) -> (AB::Expr, AB::Expr) {
    let mut is_valid = AB::Expr::ZERO;
    let mut bit_shift = AB::Expr::ZERO;
    for (i, &marker) in bit_shift_marker.iter().enumerate() {
        builder.assert_bool(marker);
        is_valid += marker.into();
        bit_shift += AB::Expr::from_usize(i) * marker;

        builder
            .when(marker)
            .assert_eq(bit_multiplier, AB::Expr::from_usize(1 << i));
    }
    builder.assert_bool(is_valid.clone());
    (is_valid, bit_shift)
}

/// Constrains the one-hot `limb_shift_marker` and returns `limb_shift`.
fn eval_limb_shift<AB: InteractionBuilder, const NUM_LIMBS: usize>(
    builder: &mut AB,
    limb_shift_marker: &[AB::Var; NUM_LIMBS],
    is_valid: AB::Expr,
) -> AB::Expr {
    let mut limb_marker_sum = AB::Expr::ZERO;
    let mut limb_shift = AB::Expr::ZERO;
    for (i, &marker) in limb_shift_marker.iter().enumerate() {
        builder.assert_bool(marker);
        limb_marker_sum += marker.into();
        limb_shift += AB::Expr::from_usize(i) * marker;
    }
    builder.when(is_valid).assert_one(limb_marker_sum);
    limb_shift
}

/// Constrains `a = b << (limb_shift * LIMB_BITS + bit_shift)`.
fn eval_left_limbs<AB: InteractionBuilder, const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    builder: &mut AB,
    a: &[AB::Var; NUM_LIMBS],
    b: &[AB::Var; NUM_LIMBS],
    bit_multiplier: AB::Var,
    bit_shift_carry: &[AB::Var; NUM_LIMBS],
    limb_shift_marker: &[AB::Var; NUM_LIMBS],
) {
    for i in 0..NUM_LIMBS {
        let mut when_limb_shift = builder.when(limb_shift_marker[i]);
        for j in 0..NUM_LIMBS {
            if j < i {
                when_limb_shift.assert_zero(a[j]);
            } else {
                let expected = if j - i == 0 {
                    AB::Expr::ZERO
                } else {
                    bit_shift_carry[j - i - 1].into()
                } + b[j - i] * bit_multiplier
                    - AB::Expr::from_usize(1 << LIMB_BITS) * bit_shift_carry[j - i];
                when_limb_shift.assert_eq(a[j], expected);
            }
        }
    }
}

/// Constrains `a = b >> (limb_shift * LIMB_BITS + bit_shift)`, filling vacated high bits with
/// `b_sign` (`0` for logical SRL, the sign bit for arithmetic SRA).
fn eval_right_limbs<AB: InteractionBuilder, const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    builder: &mut AB,
    a: &[AB::Var; NUM_LIMBS],
    b: &[AB::Var; NUM_LIMBS],
    bit_multiplier: AB::Var,
    b_sign: AB::Expr,
    bit_shift_carry: &[AB::Var; NUM_LIMBS],
    limb_shift_marker: &[AB::Var; NUM_LIMBS],
) {
    for i in 0..NUM_LIMBS {
        let mut when_limb_shift = builder.when(limb_shift_marker[i]);
        for j in 0..NUM_LIMBS {
            if j + i > NUM_LIMBS - 1 {
                when_limb_shift.assert_eq(
                    a[j].into(),
                    b_sign.clone() * AB::F::from_usize((1 << LIMB_BITS) - 1),
                );
            } else {
                let high = if j + i == NUM_LIMBS - 1 {
                    b_sign.clone() * (bit_multiplier - AB::F::ONE)
                } else {
                    bit_shift_carry[j + i + 1].into()
                };
                let expected =
                    high * AB::F::from_usize(1 << LIMB_BITS) + (b[j + i] - bit_shift_carry[j + i]);
                when_limb_shift.assert_eq(a[j] * bit_multiplier, expected);
            }
        }
    }
}

/// Range checks shared by all three chips: the bit/limb shift decomposition of `c[0]`, the byte
/// bounds on `a`/`b`/`c`, and the per-limb bit-shift carries.
#[allow(clippy::too_many_arguments)]
fn eval_ranges<AB: InteractionBuilder, const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    bitwise_lookup_bus: BitwiseOperationLookupBus,
    range_bus: VariableRangeCheckerBus,
    builder: &mut AB,
    a: &[AB::Var; NUM_LIMBS],
    b: &[AB::Var; NUM_LIMBS],
    c: &[AB::Var; NUM_LIMBS],
    bit_shift_carry: &[AB::Var; NUM_LIMBS],
    limb_shift: AB::Expr,
    bit_shift: AB::Expr,
    is_valid: AB::Expr,
) {
    // Check that `c[0] = limb_shift * LIMB_BITS + bit_shift` modulo `NUM_LIMBS * LIMB_BITS`.
    let num_bits = AB::F::from_usize(NUM_LIMBS * LIMB_BITS);
    range_bus
        .range_check(
            (c[0] - limb_shift * AB::F::from_usize(LIMB_BITS) - bit_shift.clone())
                * num_bits.inverse(),
            LIMB_BITS - ((NUM_LIMBS * LIMB_BITS) as u32).ilog2() as usize,
        )
        .eval(builder, is_valid.clone());

    for i in 0..(NUM_LIMBS / 2) {
        bitwise_lookup_bus
            .send_range(a[i * 2], a[i * 2 + 1])
            .eval(builder, is_valid.clone());
    }

    // Memory bus checks only packed u16 values; these byte limbs need separate bounds.
    for i in 0..(NUM_LIMBS / 2) {
        bitwise_lookup_bus
            .send_range(b[i * 2], b[i * 2 + 1])
            .eval(builder, is_valid.clone());
        bitwise_lookup_bus
            .send_range(c[i * 2], c[i * 2 + 1])
            .eval(builder, is_valid.clone());
    }

    for &carry in bit_shift_carry {
        range_bus
            .send(carry, bit_shift.clone())
            .eval(builder, is_valid.clone());
    }
}
