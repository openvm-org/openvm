use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::{air::LongMultiplicationAir, columns::LongMultiplicationCols};
use crate::long_multiplication::num_limbs;

impl LongMultiplicationAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: LongMultiplicationCols<AB::Var>,
    ) {
        let num_limbs = num_limbs(self.arg_size, self.limb_size);
        for z in local.z_limbs {
            // TODO: replace with a more optimal range check once our range checker supports it
            builder.push_send(self.bus_index, vec![z], local.rcv_count);
            builder.push_send(
                self.bus_index,
                vec![z + AB::Expr::from_canonical_usize((num_limbs - 1) << self.limb_size)],
                local.rcv_count,
            );
        }
        for c in local.carry {
            builder.push_send(self.bus_index, vec![c], local.rcv_count);
        }
    }
}
