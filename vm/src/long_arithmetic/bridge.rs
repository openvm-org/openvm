use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::{air::LongAdditionAir, columns::LongAdditionCols};

impl<const ARG_SIZE: usize, const LIMB_SIZE: usize> LongAdditionAir<ARG_SIZE, LIMB_SIZE> {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: LongAdditionCols<ARG_SIZE, LIMB_SIZE, AB::Var>,
    ) {
        for z in local.z_limbs {
            builder.push_send(self.bus_index, vec![z], AB::F::one());
        }
    }
}
