use std::borrow::Borrow;

use afs_primitives::range_tuple::bus::RangeTupleCheckerBus;
use afs_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use crate::{
    arch::bridge::ExecutionBridge, memory::offline_checker::MemoryBridge,
    uint_multiplication::columns::UintMultiplicationCols,
};

#[derive(Clone, Debug)]
pub struct UintMultiplicationAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    pub bus: RangeTupleCheckerBus,

    pub(super) offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for UintMultiplicationAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        UintMultiplicationCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for UintMultiplicationAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB: InteractionBuilder + AirBuilder, const NUM_LIMBS: usize, const LIMB_BITS: usize> Air<AB>
    for UintMultiplicationAir<NUM_LIMBS, LIMB_BITS>
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);

        let UintMultiplicationCols::<_, NUM_LIMBS, LIMB_BITS> { io, aux } = (*local).borrow();
        builder.assert_bool(aux.is_valid);

        let x_limbs = &io.x.data;
        let y_limbs = &io.y.data;
        let z_limbs = &io.z.data;
        let carry_limbs = &aux.carry;

        for i in 0..NUM_LIMBS {
            let lhs = (0..=i).fold(
                if i > 0 {
                    carry_limbs[i - 1].into()
                } else {
                    AB::Expr::zero()
                },
                |acc, j| acc + (x_limbs[j] * y_limbs[i - j]),
            );
            let rhs =
                z_limbs[i] + (carry_limbs[i] * AB::Expr::from_canonical_usize(1 << LIMB_BITS));
            builder.assert_eq(lhs, rhs);
        }

        self.eval_interactions(builder, io, aux);
    }
}
