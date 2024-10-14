use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::{air::ModularMultDivAir, ModularMultDivAuxCols, ModularMultDivIoCols};

impl<const CARRY_LIMBS: usize, const NUM_LIMBS: usize, const LIMB_SIZE: usize>
    ModularMultDivAir<CARRY_LIMBS, NUM_LIMBS, LIMB_SIZE>
{
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: &ModularMultDivIoCols<AB::Var, NUM_LIMBS>,
        aux: &ModularMultDivAuxCols<AB::Var, CARRY_LIMBS, NUM_LIMBS>,
        expected_opcode: AB::Expr,
    ) {
        let timestamp: AB::Var = io.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::Expr::from_canonical_usize(timestamp_delta - 1)
        };

        self.memory_bridge
            .read_from_cols(
                io.x.address.clone(),
                timestamp_pp(),
                &aux.read_x_aux_cols.address,
            )
            .eval(builder, aux.is_valid);

        self.memory_bridge
            .read_from_cols(io.x.data.clone(), timestamp_pp(), &aux.read_x_aux_cols.data)
            .eval(builder, aux.is_valid);

        self.memory_bridge
            .read_from_cols(
                io.y.address.clone(),
                timestamp_pp(),
                &aux.read_y_aux_cols.address,
            )
            .eval(builder, aux.is_valid);

        self.memory_bridge
            .read_from_cols(io.y.data.clone(), timestamp_pp(), &aux.read_y_aux_cols.data)
            .eval(builder, aux.is_valid);

        self.memory_bridge
            .read_from_cols(
                io.z.address.clone(),
                timestamp_pp(),
                &aux.write_z_aux_cols.address,
            )
            .eval(builder, aux.is_valid);

        self.memory_bridge
            .write_from_cols(
                io.z.data.clone(),
                timestamp_pp(),
                &aux.write_z_aux_cols.data,
            )
            .eval(builder, aux.is_valid);

        self.execution_bridge
            .execute_and_increment_pc(
                expected_opcode + AB::Expr::from_canonical_usize(self.offset),
                [
                    io.z.address.address,
                    io.x.address.address,
                    io.y.address.address,
                    io.x.address.address_space,
                    io.x.data.address_space,
                ],
                io.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
            )
            .eval(builder, aux.is_valid);
    }
}
