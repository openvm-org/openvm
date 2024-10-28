use std::{borrow::Borrow, iter::zip};

use ax_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupBus, utils, var_range::VariableRangeCheckerBus,
};
use ax_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::columns::ShiftCols;
use crate::{
    arch::{instructions::U256Opcode, ExecutionBridge},
    system::memory::offline_checker::MemoryBridge,
};

#[derive(Clone, Copy, Debug)]
pub struct ShiftCoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
    pub range_bus: VariableRangeCheckerBus,

    pub(super) offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> PartitionedBaseAir<F>
    for ShiftCoreAir<NUM_LIMBS, LIMB_BITS>
{
}
impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for ShiftCoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        ShiftCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for ShiftCoreAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB: InteractionBuilder + AirBuilder, const NUM_LIMBS: usize, const LIMB_BITS: usize> Air<AB>
    for ShiftCoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);

        let ShiftCols::<_, NUM_LIMBS, LIMB_BITS> { io, aux } = (*local).borrow();
        builder.assert_bool(aux.is_valid);

        // Constrain that flags are valid.
        let flags = [
            aux.opcode_sll_flag,
            aux.opcode_srl_flag,
            aux.opcode_sra_flag,
        ];
        for flag in flags {
            builder.assert_bool(flag);
        }

        builder.assert_eq(
            aux.is_valid,
            flags
                .iter()
                .fold(AB::Expr::zero(), |acc, &flag| acc + flag.into()),
        );

        let x_limbs = &io.x.data;
        let y_limbs = &io.y.data;
        let z_limbs = &io.z.data;
        let right_shift = aux.opcode_srl_flag + aux.opcode_sra_flag;

        // Constrain that bit_shift, bit_multiplier are correct, i.e. that bit_multiplier =
        // 1 << bit_shift. We check that bit_shift is correct below if y < NUM_LIMBS * LIMB_BITS,
        // otherwise we don't really care what its value is. Note that bit_shift < LIMB_BITS is
        // constrained in bridge.rs via the range checker.
        builder
            .when(aux.opcode_sll_flag)
            .assert_zero(aux.bit_multiplier_right);
        builder
            .when(right_shift.clone())
            .assert_zero(aux.bit_multiplier_left);

        for i in 0..LIMB_BITS {
            let mut when_bit_shift = builder.when(aux.bit_shift_marker[i]);
            when_bit_shift.assert_eq(aux.bit_shift, AB::F::from_canonical_usize(i));
            when_bit_shift
                .when(aux.opcode_sll_flag)
                .assert_eq(aux.bit_multiplier_left, AB::F::from_canonical_usize(1 << i));
            when_bit_shift.when(right_shift.clone()).assert_eq(
                aux.bit_multiplier_right,
                AB::F::from_canonical_usize(1 << i),
            );
        }

        builder.assert_bool(aux.x_sign);
        builder
            .when(utils::not(aux.opcode_sra_flag))
            .assert_zero(aux.x_sign);

        let mut marker_sum = AB::Expr::zero();

        // Check that z[i] = x[i] <</>> y[i] both on the bit and limb shift level if y <
        // NUM_LIMBS * LIMB_BITS.
        for i in 0..NUM_LIMBS {
            marker_sum += aux.limb_shift_marker[i].into();
            builder.assert_bool(aux.limb_shift_marker[i]);

            let mut when_limb_shift = builder.when(aux.limb_shift_marker[i]);
            when_limb_shift.assert_eq(
                y_limbs[1] * AB::F::from_canonical_usize(1 << LIMB_BITS) + y_limbs[0]
                    - aux.bit_shift,
                AB::F::from_canonical_usize(i * LIMB_BITS),
            );

            for j in 0..NUM_LIMBS {
                // SLL constraints
                if j < i {
                    when_limb_shift.assert_zero(z_limbs[j] * aux.opcode_sll_flag);
                } else {
                    let expected_z_left = if j - i == 0 {
                        AB::Expr::zero()
                    } else {
                        aux.bit_shift_carry[j - i - 1].into() * aux.opcode_sll_flag
                    } + x_limbs[j - i] * aux.bit_multiplier_left
                        - AB::Expr::from_canonical_usize(1 << LIMB_BITS)
                            * aux.bit_shift_carry[j - i]
                            * aux.opcode_sll_flag;
                    when_limb_shift.assert_eq(z_limbs[j] * aux.opcode_sll_flag, expected_z_left);
                }

                // SRL and SRA constraints. Combining with above would require an additional column.
                if j + i > NUM_LIMBS - 1 {
                    when_limb_shift.assert_eq(
                        z_limbs[j] * right_shift.clone(),
                        aux.x_sign * AB::F::from_canonical_usize((1 << LIMB_BITS) - 1),
                    );
                } else {
                    let expected_z_right = if j + i == NUM_LIMBS - 1 {
                        aux.x_sign * (aux.bit_multiplier_right - AB::F::one())
                    } else {
                        aux.bit_shift_carry[j + i + 1].into() * right_shift.clone()
                    } * AB::F::from_canonical_usize(1 << LIMB_BITS)
                        + right_shift.clone() * (x_limbs[j + i] - aux.bit_shift_carry[j + i]);
                    when_limb_shift
                        .assert_eq(z_limbs[j] * aux.bit_multiplier_right, expected_z_right);
                }

                // Ensure y is defined entirely within y[0] and y[1] if limb shifting
                if j > 1 {
                    when_limb_shift.assert_zero(y_limbs[j]);
                }
            }
        }

        // If the shift is larger than the number of bits, check that each limb of z is filled
        for z in z_limbs {
            builder
                .when(AB::Expr::one() - marker_sum.clone())
                .assert_eq(
                    *z,
                    aux.x_sign * AB::F::from_canonical_usize((1 << LIMB_BITS) - 1),
                );
        }

        let expected_opcode = zip(flags, U256Opcode::shift_opcodes())
            .fold(AB::Expr::zero(), |acc, (flag, opcode)| {
                acc + flag * AB::Expr::from_canonical_u8(opcode as u8)
            });

        self.eval_interactions(builder, io, aux, expected_opcode);
    }
}
