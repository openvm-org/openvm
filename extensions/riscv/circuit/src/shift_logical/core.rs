use std::borrow::Borrow;

use openvm_circuit::arch::*;
use openvm_circuit_primitives::{
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_riscv_transpiler::ShiftOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing},
    BaseAirWithPublicValues,
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Clone, Copy, Debug)]
pub struct ShiftLogicalCoreCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],

    pub opcode_sll_flag: T,

    // bit_multiplier = 2^bit_shift (scaled by the active opcode flag, 0 otherwise)
    pub bit_multiplier_left: T,
    // carry_multiplier = 2^(LIMB_BITS - bit_shift) (scaled by the active opcode flag).
    // Used to position the part of each limb that crosses the limb boundary without forming a
    // product that exceeds 2^LIMB_BITS (which would alias the field modulus for u16 limbs).
    pub carry_multiplier_left: T,

    // Boolean columns that are 1 exactly at the index of the bit/limb shift amount
    pub bit_shift_marker: [T; LIMB_BITS],
    pub limb_shift_marker: [T; NUM_LIMBS],

    // Decomposition of each b[k] into the part that crosses into the next limb (`carry`) and the
    // part that stays (`aux`):
    //   SLL: b[k] = aux[k] + carry[k] * 2^(LIMB_BITS - bit_shift), carry = high bit_shift bits
    //   SRL: b[k] = carry[k] + aux[k] * 2^bit_shift,               carry = low  bit_shift bits
    // `carry` is range checked to bit_shift bits, `aux` to LIMB_BITS - bit_shift bits.
    pub bit_shift_carry: [T; NUM_LIMBS],
    pub bit_shift_aux: [T; NUM_LIMBS],
}

/// Logical shift AIR (SLL/SRL) over u16 limbs.
///
/// To stay sound at `LIMB_BITS = 16`, each `b` limb is split into `carry`/`aux` parts and
/// recombined additively so every constraint term stays below BabyBear's modulus.
///
/// Note: when the shift amount from operand is greater than the number of bits, only shift
/// `shift_amount % num_bits` bits. This matches the RISC-V specs for SLL/SRL.
#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(ShiftLogicalCoreCols<u8, NUM_LIMBS, LIMB_BITS>)]
pub struct ShiftLogicalCoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub range_bus: VariableRangeCheckerBus,
    pub offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for ShiftLogicalCoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        ShiftLogicalCoreCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}
impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for ShiftLogicalCoreAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for ShiftLogicalCoreAir<NUM_LIMBS, LIMB_BITS>
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
        let cols: &ShiftLogicalCoreCols<_, NUM_LIMBS, LIMB_BITS> = local_core.borrow();
        builder.assert_bool(cols.opcode_sll_flag);
        let opcode_sll_flag: AB::Expr = cols.opcode_sll_flag.into();

        let a = &cols.a;
        let b = &cols.b;
        let c = &cols.c;

        // Constrain that bit_shift and the bit/carry multipliers are correct. The marker sum is 1
        // on valid rows and 0 on padding rows.
        let mut bit_marker_sum = AB::Expr::ZERO;
        let mut bit_shift = AB::Expr::ZERO;
        let mut bit_multiplier = AB::Expr::ZERO;
        let mut carry_multiplier = AB::Expr::ZERO;

        for i in 0..LIMB_BITS {
            builder.assert_bool(cols.bit_shift_marker[i]);
            let marker: AB::Expr = cols.bit_shift_marker[i].into();
            bit_marker_sum += marker.clone();
            bit_shift += AB::Expr::from_usize(i) * marker.clone();
            bit_multiplier += AB::Expr::from_usize(1 << i) * marker.clone();
            carry_multiplier += AB::Expr::from_usize(1 << (LIMB_BITS - i)) * marker;
        }
        builder.assert_bool(bit_marker_sum.clone());
        let is_valid = bit_marker_sum;
        // A valid row is SLL iff `opcode_sll_flag` is set; otherwise it is SRL. Booleanity of the
        // derived SRL flag forces the SLL flag to be zero on padding rows.
        let opcode_srl_flag = is_valid.clone() - opcode_sll_flag.clone();
        builder.assert_bool(opcode_srl_flag.clone());
        builder.assert_eq(
            cols.bit_multiplier_left,
            bit_multiplier.clone() * opcode_sll_flag.clone(),
        );
        builder.assert_eq(
            cols.carry_multiplier_left,
            carry_multiplier.clone() * opcode_sll_flag.clone(),
        );

        // Decompose each b[k] into carry/aux parts. Multiplying the active opcode flag into each
        // constraint makes it vacuous for the inactive opcode.
        for (k, &b_limb) in b.iter().enumerate() {
            // SLL: b[k] = aux[k] + carry[k] * 2^(LIMB_BITS - bit_shift)
            builder.assert_eq(
                b_limb * cols.opcode_sll_flag,
                cols.bit_shift_aux[k] * cols.opcode_sll_flag
                    + cols.bit_shift_carry[k] * cols.carry_multiplier_left,
            );
            // SRL: b[k] = carry[k] + aux[k] * 2^bit_shift
            builder.assert_eq(
                b_limb * opcode_srl_flag.clone(),
                cols.bit_shift_carry[k] * opcode_srl_flag.clone()
                    + cols.bit_shift_aux[k] * (bit_multiplier.clone() - cols.bit_multiplier_left),
            );
        }

        // Check that a[i] = b[i] <</>> c[i] both on the bit and limb shift level if c <
        // NUM_LIMBS * LIMB_BITS. Each output limb is recombined additively from the (already
        // range-checked) carry/aux parts, so the result is automatically in [0, 2^LIMB_BITS).
        let mut limb_marker_sum = AB::Expr::ZERO;
        let mut limb_shift = AB::Expr::ZERO;
        for i in 0..NUM_LIMBS {
            builder.assert_bool(cols.limb_shift_marker[i]);
            limb_marker_sum += cols.limb_shift_marker[i].into();
            limb_shift += AB::Expr::from_usize(i) * cols.limb_shift_marker[i];

            let mut when_limb_shift = builder.when(cols.limb_shift_marker[i]);

            for (j, &a_limb) in a.iter().enumerate() {
                // SLL: a[j] = aux[j-i] * 2^bit_shift + carry[j-i-1]
                if j < i {
                    when_limb_shift.assert_zero(a_limb * cols.opcode_sll_flag);
                } else {
                    let carry_in = if j - i == 0 {
                        AB::Expr::ZERO
                    } else {
                        cols.bit_shift_carry[j - i - 1].into() * cols.opcode_sll_flag
                    };
                    when_limb_shift.assert_eq(
                        a_limb * cols.opcode_sll_flag,
                        cols.bit_shift_aux[j - i] * cols.bit_multiplier_left + carry_in,
                    );
                }

                // SRL: a[j] = aux[j+i] + carry[j+i+1] * 2^(LIMB_BITS - bit_shift)
                if j + i > NUM_LIMBS - 1 {
                    when_limb_shift.assert_zero(a_limb * opcode_srl_flag.clone());
                } else {
                    let carry_in = if j + i == NUM_LIMBS - 1 {
                        AB::Expr::ZERO
                    } else {
                        cols.bit_shift_carry[j + i + 1].into()
                            * (carry_multiplier.clone() - cols.carry_multiplier_left)
                    };
                    when_limb_shift.assert_eq(
                        a_limb * opcode_srl_flag.clone(),
                        cols.bit_shift_aux[j + i] * opcode_srl_flag.clone() + carry_in,
                    );
                }
            }
        }
        builder.assert_eq(limb_marker_sum, is_valid.clone());

        // Check that bit_shift and limb_shift are correct.
        let num_bits = AB::F::from_usize(NUM_LIMBS * LIMB_BITS);
        self.range_bus
            .range_check(
                (c[0] - limb_shift * AB::F::from_usize(LIMB_BITS) - bit_shift.clone())
                    * num_bits.inverse(),
                LIMB_BITS - ((NUM_LIMBS * LIMB_BITS) as u32).ilog2() as usize,
            )
            .eval(builder, is_valid.clone());

        // Range check the carry/aux decomposition of each b limb. b and c arrive range checked to
        // [0, 2^LIMB_BITS) from the u16 memory bus, so no further bounds on b/c are needed; a is
        // bounded implicitly by the additive recombination above.
        let aux_bits = AB::Expr::from_usize(LIMB_BITS) - bit_shift.clone();
        for k in 0..NUM_LIMBS {
            self.range_bus
                .send(cols.bit_shift_carry[k], bit_shift.clone())
                .eval(builder, is_valid.clone());
            self.range_bus
                .send(cols.bit_shift_aux[k], aux_bits.clone())
                .eval(builder, is_valid.clone());
        }

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            [
                (opcode_sll_flag, ShiftOpcode::SLL),
                (opcode_srl_flag, ShiftOpcode::SRL),
            ]
            .iter()
            .fold(AB::Expr::ZERO, |acc, (flag, opcode)| {
                acc + flag.clone() * AB::Expr::from_u8(*opcode as u8)
            }),
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

#[derive(Clone, Copy, derive_new::new)]
pub struct ShiftLogicalCoreExecutor<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub offset: usize,
}

#[derive(Clone, derive_new::new)]
pub struct ShiftLogicalFiller {
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

// Returns (result, limb_shift, bit_shift)
#[inline(always)]
pub(crate) fn run_shift_logical<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    opcode: ShiftOpcode,
    x: &[u16; NUM_LIMBS],
    y: &[u16; NUM_LIMBS],
) -> ([u16; NUM_LIMBS], usize, usize) {
    match opcode {
        ShiftOpcode::SLL => run_shift_left::<NUM_LIMBS, LIMB_BITS>(x, y),
        // SRL
        _ => run_shift_right_logical::<NUM_LIMBS, LIMB_BITS>(x, y),
    }
}

#[inline(always)]
fn run_shift_left<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u16; NUM_LIMBS],
    y: &[u16; NUM_LIMBS],
) -> ([u16; NUM_LIMBS], usize, usize) {
    let mut result = [0u16; NUM_LIMBS];

    let (limb_shift, bit_shift) = get_shift_u16::<NUM_LIMBS, LIMB_BITS>(y);

    for i in limb_shift..NUM_LIMBS {
        result[i] = if i > limb_shift {
            (((x[i - limb_shift] as u32) << bit_shift)
                | ((x[i - limb_shift - 1] as u32) >> (LIMB_BITS - bit_shift)))
                % (1u32 << LIMB_BITS)
        } else {
            ((x[i - limb_shift] as u32) << bit_shift) % (1u32 << LIMB_BITS)
        } as u16;
    }
    (result, limb_shift, bit_shift)
}

#[inline(always)]
fn run_shift_right_logical<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u16; NUM_LIMBS],
    y: &[u16; NUM_LIMBS],
) -> ([u16; NUM_LIMBS], usize, usize) {
    let mut result = [0u16; NUM_LIMBS];

    let (limb_shift, bit_shift) = get_shift_u16::<NUM_LIMBS, LIMB_BITS>(y);

    for i in 0..(NUM_LIMBS - limb_shift) {
        let res = if i + limb_shift + 1 < NUM_LIMBS {
            (((x[i + limb_shift] as u32) >> bit_shift)
                | ((x[i + limb_shift + 1] as u32) << (LIMB_BITS - bit_shift)))
                % (1u32 << LIMB_BITS)
        } else {
            ((x[i + limb_shift] as u32) >> bit_shift) % (1u32 << LIMB_BITS)
        };
        result[i] = res as u16;
    }
    (result, limb_shift, bit_shift)
}

#[inline(always)]
fn get_shift_u16<const NUM_LIMBS: usize, const LIMB_BITS: usize>(y: &[u16]) -> (usize, usize) {
    debug_assert!(NUM_LIMBS * LIMB_BITS <= (1 << LIMB_BITS));
    // We assume `NUM_LIMBS * LIMB_BITS <= 2^LIMB_BITS` so the shift is defined
    // entirely in y[0].
    let shift = (y[0] as usize) % (NUM_LIMBS * LIMB_BITS);
    (shift / LIMB_BITS, shift % LIMB_BITS)
}
