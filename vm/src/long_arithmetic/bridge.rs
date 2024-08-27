use afs_stark_backend::interaction::InteractionBuilder;

use super::{air::LongArithmeticAir, columns::LongArithmeticCols};

impl<const ARG_SIZE: usize, const LIMB_SIZE: usize> LongArithmeticAir<ARG_SIZE, LIMB_SIZE> {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: LongArithmeticCols<ARG_SIZE, LIMB_SIZE, AB::Var>,
    ) {
        for z in local.io.z_limbs {
            builder.push_send(
                self.bus_index,
                vec![z],
                local.io.rcv_count
                    * (local.aux.opcode_add_flag
                        + local.aux.opcode_sub_flag
                        + local.aux.opcode_lt_flag),
            );
        }
    }
}
