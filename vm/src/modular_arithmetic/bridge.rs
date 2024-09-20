use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::{
    columns::{ModularArithmeticAuxCols, ModularArithmeticIoCols},
    ModularArithmeticAirVariant, ModularArithmeticVmAir,
};

impl ModularArithmeticVmAir<ModularArithmeticAirVariant> {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: ModularArithmeticIoCols<AB::Var>,
        aux: ModularArithmeticAuxCols<AB::Var>,
    ) {
        let mut timestamp_delta = AB::Expr::zero();
        let timestamp: AB::Expr = io.from_state.timestamp.into();

        // Interaction with program
        self.program_bus.send_instruction(
            builder,
            io.from_state.pc,
            aux.opcode,
            [
                io.z.address.address,
                io.x.address.address,
                io.y.address.address,
                io.x.address.address_space,
                io.x.data.address_space,
            ],
            aux.is_valid,
        );

        self.memory_bridge
            .read_from_cols(
                io.x.address.clone(),
                timestamp.clone() + timestamp_delta.clone(),
                &aux.read_x_aux_cols.address,
            )
            .eval(builder, aux.is_valid);
        timestamp_delta += AB::Expr::one();
        self.memory_bridge
            .read_from_cols(
                io.x.data.clone(),
                timestamp.clone() + timestamp_delta.clone(),
                &aux.read_x_aux_cols.data,
            )
            .eval(builder, aux.is_valid);
        timestamp_delta += AB::Expr::one();

        self.memory_bridge
            .read_from_cols(
                io.y.address.clone(),
                timestamp.clone() + timestamp_delta.clone(),
                &aux.read_y_aux_cols.address,
            )
            .eval(builder, aux.is_valid);
        timestamp_delta += AB::Expr::one();
        self.memory_bridge
            .read_from_cols(
                io.y.data.clone(),
                timestamp.clone() + timestamp_delta.clone(),
                &aux.read_y_aux_cols.data,
            )
            .eval(builder, aux.is_valid);
        timestamp_delta += AB::Expr::one();

        self.memory_bridge
            .read_from_cols(
                io.z.address.clone(),
                timestamp.clone() + timestamp_delta.clone(),
                &aux.write_z_aux_cols.address,
            )
            .eval(builder, aux.is_valid);
        timestamp_delta += AB::Expr::one();
        self.memory_bridge
            .write_from_cols(
                io.z.data.clone(),
                timestamp.clone() + timestamp_delta.clone(),
                &aux.write_z_aux_cols.data,
            )
            .eval(builder, aux.is_valid);
        timestamp_delta += AB::Expr::one();

        self.execution_bus.execute_increment_pc(
            builder,
            aux.is_valid,
            io.from_state.map(Into::into),
            timestamp_delta,
        );
    }
}
