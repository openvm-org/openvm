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

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Clone, Copy, Debug)]
pub struct ShiftRightArithmeticImmCoreCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],

    pub b_sign: T,

    pub bit_shift_marker: [T; LIMB_BITS],
    pub limb_shift_marker: [T; NUM_LIMBS],
    pub bit_shift_carry: [T; NUM_LIMBS],
    pub bit_shift_aux: [T; NUM_LIMBS],
}

/// Arithmetic shift-right-by-immediate AIR over u16 limbs.
///
/// The marker columns uniquely encode a shift in `0..NUM_LIMBS * LIMB_BITS`; the execution
/// bridge binds that encoding directly to the instruction immediate. Consequently this core
/// needs neither immediate limbs nor the quotient range check used by the register SRA core.
#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(ShiftRightArithmeticImmCoreCols<u8, NUM_LIMBS, LIMB_BITS>)]
pub struct ShiftRightArithmeticImmCoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub range_bus: VariableRangeCheckerBus,
    pub offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for ShiftRightArithmeticImmCoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        ShiftRightArithmeticImmCoreCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for ShiftRightArithmeticImmCoreAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for ShiftRightArithmeticImmCoreAir<NUM_LIMBS, LIMB_BITS>
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
        let cols: &ShiftRightArithmeticImmCoreCols<_, NUM_LIMBS, LIMB_BITS> = local_core.borrow();
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

        for (k, &b_limb) in cols.b.iter().enumerate() {
            builder.assert_eq(
                b_limb,
                cols.bit_shift_carry[k] + cols.bit_shift_aux[k] * bit_multiplier.clone(),
            );
        }

        let mut limb_marker_sum = AB::Expr::ZERO;
        let mut limb_shift = AB::Expr::ZERO;
        for i in 0..NUM_LIMBS {
            builder.assert_bool(cols.limb_shift_marker[i]);
            limb_marker_sum += cols.limb_shift_marker[i].into();
            limb_shift += AB::Expr::from_usize(i) * cols.limb_shift_marker[i];

            let mut when_limb_shift = builder.when(cols.limb_shift_marker[i]);
            for (j, &a_limb) in cols.a.iter().enumerate() {
                if j + i > NUM_LIMBS - 1 {
                    when_limb_shift.assert_eq(
                        a_limb,
                        cols.b_sign * AB::F::from_usize((1 << LIMB_BITS) - 1),
                    );
                } else {
                    let carry_in = if j + i == NUM_LIMBS - 1 {
                        (AB::Expr::from_usize(1 << LIMB_BITS) - carry_multiplier.clone())
                            * cols.b_sign
                    } else {
                        carry_multiplier.clone() * cols.bit_shift_carry[j + i + 1]
                    };
                    when_limb_shift.assert_eq(a_limb, carry_in + cols.bit_shift_aux[j + i]);
                }
            }
        }
        builder.assert_eq(limb_marker_sum, is_valid.clone());

        builder.assert_bool(cols.b_sign);
        self.range_bus
            .range_check(
                cols.b[NUM_LIMBS - 1] - cols.b_sign * AB::F::from_u32(1 << (LIMB_BITS - 1)),
                LIMB_BITS - 1,
            )
            .eval(builder, is_valid.clone());

        let aux_bits = AB::Expr::from_usize(LIMB_BITS) - bit_shift.clone();
        for k in 0..NUM_LIMBS {
            self.range_bus
                .send(cols.bit_shift_carry[k], bit_shift.clone())
                .eval(builder, is_valid.clone());
            self.range_bus
                .send(cols.bit_shift_aux[k], aux_bits.clone())
                .eval(builder, is_valid.clone());
        }

        let immediate = limb_shift * AB::Expr::from_usize(LIMB_BITS) + bit_shift;
        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            AB::Expr::from_usize(ShiftImmOpcode::SRAI as usize),
        );

        AdapterAirContext {
            to_pc: None,
            reads: [cols.b.map(Into::into)].into(),
            writes: [cols.a.map(Into::into)].into(),
            instruction: ImmInstruction {
                is_valid,
                opcode: expected_opcode,
                immediate,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        self.offset
    }
}

#[derive(Clone, Copy, derive_new::new)]
pub struct ShiftRightArithmeticImmExecutor<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub offset: usize,
}

#[derive(Clone, derive_new::new)]
pub struct ShiftRightArithmeticImmFiller {
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}
