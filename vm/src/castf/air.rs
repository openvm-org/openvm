use std::borrow::Borrow;

use afs_primitives::var_range::bus::VariableRangeCheckerBus;
use afs_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use super::columns::CastFCols;
use crate::{arch::bridge::ExecutionBridge, memory::offline_checker::MemoryBridge};

// LIMB_SIZE is the size of the limbs in bits.
pub(crate) const LIMB_SIZE: usize = 8;
// the final limb has only 6 bits
pub(crate) const FINAL_LIMB_SIZE: usize = 6;

// AIR for casting one u30 number into three u8 and one u6 numbers
#[derive(Copy, Clone, Debug)]
pub struct CastFAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,

    pub bus: VariableRangeCheckerBus, // to communicate with the range checker that checks that all limbs are < 2^LIMB_SIZE
}

impl<F: Field> BaseAirWithPublicValues<F> for CastFAir {}
impl<F: Field> PartitionedBaseAir<F> for CastFAir {}
impl<F: Field> BaseAir<F> for CastFAir {
    fn width(&self) -> usize {
        CastFCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> Air<AB> for CastFAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local_cols: &CastFCols<AB::Var> = (*local).borrow();
        builder.assert_bool(local_cols.aux.is_valid);

        self.eval_interactions(builder, &local_cols.io, &local_cols.aux);
    }
}
