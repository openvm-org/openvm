use std::borrow::Borrow;

use openvm_circuit::arch::*;
use openvm_circuit_primitives::{
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_riscv_transpiler::ShiftImmOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing},
    BaseAirWithPublicValues,
};

/// Core columns for logical shifts with an immediate shift amount.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Clone, Copy, Debug)]
pub struct ShiftLogicalImmCoreCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],

    pub opcode_sll_flag: T,

    pub bit_multiplier_left: T,
    pub carry_multiplier_left: T,

    pub bit_shift_marker: [T; LIMB_BITS],
    pub limb_shift_marker: [T; NUM_LIMBS],

    pub bit_shift_carry: [T; NUM_LIMBS],
    pub bit_shift_aux: [T; NUM_LIMBS],
}

/// Logical shift-by-immediate AIR (SLLI/SRLI) over u16 limbs.
///
/// The marker columns uniquely encode `shamt`, and the execution bridge binds
/// `limb_shift * LIMB_BITS + bit_shift` to the immediate operand.
#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(ShiftLogicalImmCoreCols<u8, NUM_LIMBS, LIMB_BITS>)]
pub struct ShiftLogicalImmCoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub range_bus: VariableRangeCheckerBus,
    pub offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for ShiftLogicalImmCoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        ShiftLogicalImmCoreCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}
impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for ShiftLogicalImmCoreAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for ShiftLogicalImmCoreAir<NUM_LIMBS, LIMB_BITS>
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
        let cols: &ShiftLogicalImmCoreCols<_, NUM_LIMBS, LIMB_BITS> = local_core.borrow();
        builder.assert_bool(cols.opcode_sll_flag);
        let opcode_sll_flag: AB::Expr = cols.opcode_sll_flag.into();

        let a = &cols.a;
        let b = &cols.b;

        // Constrain that bit_shift and the (bit/carry) multipliers are correct.
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

        // Decompose each b[k] into carry/aux parts (see ShiftLogicalCoreAir).
        for (k, &b_limb) in b.iter().enumerate() {
            builder.assert_eq(
                b_limb * cols.opcode_sll_flag,
                cols.bit_shift_aux[k] * cols.opcode_sll_flag
                    + cols.bit_shift_carry[k] * cols.carry_multiplier_left,
            );
            builder.assert_eq(
                b_limb * opcode_srl_flag.clone(),
                cols.bit_shift_carry[k] * opcode_srl_flag.clone()
                    + cols.bit_shift_aux[k] * (bit_multiplier.clone() - cols.bit_multiplier_left),
            );
        }

        // Check that a[i] = b[i] <</>> shamt both on the bit and limb shift level.
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

        // The immediate operand is exactly limb_shift * LIMB_BITS + bit_shift; both parts are
        // bounded by the marker-sum constraints, so no range check is needed.
        let imm = limb_shift * AB::Expr::from_usize(LIMB_BITS) + bit_shift.clone();

        // Range check the carry/aux decomposition of each b limb.
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
                (opcode_sll_flag, ShiftImmOpcode::SLLI as usize),
                (opcode_srl_flag, ShiftImmOpcode::SRLI as usize),
            ]
            .iter()
            .fold(AB::Expr::ZERO, |acc, (flag, opcode)| {
                acc + flag.clone() * AB::Expr::from_usize(*opcode)
            }),
        );

        AdapterAirContext {
            to_pc: None,
            reads: [cols.b.map(Into::into)].into(),
            writes: [cols.a.map(Into::into)].into(),
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
pub struct ShiftLogicalImmExecutor<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub offset: usize,
}

#[derive(Clone, derive_new::new)]
pub struct ShiftLogicalImmFiller<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}
