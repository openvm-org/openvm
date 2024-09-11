use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::{
    columns::{ModularArithmeticAuxCols, ModularArithmeticIoCols},
    ModularArithmeticAirVariant, ModularArithmeticVmAir,
};
use crate::{arch::columns::InstructionCols, memory::MemoryAddress};

impl ModularArithmeticVmAir<ModularArithmeticAirVariant> {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: ModularArithmeticIoCols<AB::Var>,
        aux: ModularArithmeticAuxCols<AB::Var>,
    ) {
        let mut timestamp_delta = AB::Expr::zero();
        let timestamp: AB::Expr = io.from_state.timestamp.into();

        self.memory_bridge
            .read(
                MemoryAddress::new(io.x_address.address_space, io.x_address.address),
                io.x_address
                    .data
                    .try_into()
                    .unwrap_or_else(|_| unreachable!()),
                timestamp.clone() + timestamp_delta.clone(),
                &aux.x_address_aux_cols,
            )
            .eval(builder, aux.is_valid);
        timestamp_delta += AB::Expr::one();

        self.memory_bridge
            .read(
                MemoryAddress::new(io.y_address.address_space, io.y_address.address),
                io.y_address
                    .data
                    .try_into()
                    .unwrap_or_else(|_| unreachable!()),
                timestamp.clone() + timestamp_delta.clone(),
                &aux.y_address_aux_cols,
            )
            .eval(builder, aux.is_valid);
        timestamp_delta += AB::Expr::one();

        self.memory_bridge
            .read(
                MemoryAddress::new(io.z_address.address_space, io.z_address.address),
                io.z_address
                    .data
                    .try_into()
                    .unwrap_or_else(|_| unreachable!()),
                timestamp.clone() + timestamp_delta.clone(),
                &aux.z_address_aux_cols,
            )
            .eval(builder, aux.is_valid);
        timestamp_delta += AB::Expr::one();

        self.memory_bridge
            .read(
                MemoryAddress::new(io.x.address_space, io.x.address),
                io.x.data.try_into().unwrap_or_else(|_| unreachable!()),
                timestamp.clone() + timestamp_delta.clone(),
                &aux.read_x_aux_cols,
            )
            .eval(builder, aux.is_valid);
        timestamp_delta += AB::Expr::one();

        self.memory_bridge
            .read(
                MemoryAddress::new(io.y.address_space, io.y.address),
                io.y.data.try_into().unwrap_or_else(|_| unreachable!()),
                timestamp.clone() + timestamp_delta.clone(),
                &aux.read_y_aux_cols,
            )
            .eval(builder, aux.is_valid);
        timestamp_delta += AB::Expr::one();

        self.memory_bridge
            .write(
                MemoryAddress::new(io.z.address_space, io.z.address),
                io.z.data.try_into().unwrap_or_else(|_| unreachable!()),
                timestamp.clone() + timestamp_delta.clone(),
                &aux.write_z_aux_cols,
            )
            .eval(builder, aux.is_valid);
        timestamp_delta += AB::Expr::one();

        self.execution_bus.execute_increment_pc(
            builder,
            aux.is_valid,
            io.from_state.map(Into::into),
            timestamp_delta,
            InstructionCols::new(
                aux.opcode,
                [
                    io.z_address.address,
                    io.x_address.address,
                    io.y_address.address,
                    io.x_address.address_space,
                    io.x.address_space,
                ],
            ),
        );
    }
}
