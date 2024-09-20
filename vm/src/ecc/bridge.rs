use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::{
    EcAddUnequalAuxCols, EcAddUnequalIoCols, EcAddUnequalVmAir, EcDoubleAuxCols, EcDoubleIoCols,
    EcDoubleVmAir,
};
use crate::arch::instructions::Opcode;

impl EcAddUnequalVmAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: EcAddUnequalIoCols<AB::Var>,
        aux: EcAddUnequalAuxCols<AB::Var>,
    ) {
        let mut timestamp_delta = AB::Expr::zero();
        let timestamp: AB::Expr = io.from_state.timestamp.into();

        self.program_bus.send_instruction(
            builder,
            io.from_state.pc,
            AB::Expr::from_canonical_u8(Opcode::SECP256K1_EC_ADD_NE as u8),
            [
                io.p3.address.address,
                io.p1.address.address,
                io.p2.address.address,
                io.p1.address.address_space,
                io.p1.data.address_space,
            ],
            aux.aux.is_valid,
        );

        self.memory_bridge
            .read_from_cols(
                io.p1.address.clone(),
                timestamp.clone() + timestamp_delta.clone(),
                &aux.read_p1_aux_cols.address,
            )
            .eval(builder, aux.aux.is_valid);
        timestamp_delta += AB::Expr::one();
        self.memory_bridge
            .read_from_cols(
                io.p1.data.clone(),
                timestamp.clone() + timestamp_delta.clone(),
                &aux.read_p1_aux_cols.data,
            )
            .eval(builder, aux.aux.is_valid);
        timestamp_delta += AB::Expr::one();

        self.memory_bridge
            .read_from_cols(
                io.p2.address.clone(),
                timestamp.clone() + timestamp_delta.clone(),
                &aux.read_p2_aux_cols.address,
            )
            .eval(builder, aux.aux.is_valid);
        timestamp_delta += AB::Expr::one();
        self.memory_bridge
            .read_from_cols(
                io.p2.data.clone(),
                timestamp.clone() + timestamp_delta.clone(),
                &aux.read_p2_aux_cols.data,
            )
            .eval(builder, aux.aux.is_valid);
        timestamp_delta += AB::Expr::one();

        self.memory_bridge
            .read_from_cols(
                io.p3.address.clone(),
                timestamp.clone() + timestamp_delta.clone(),
                &aux.write_p3_aux_cols.address,
            )
            .eval(builder, aux.aux.is_valid);
        timestamp_delta += AB::Expr::one();
        self.memory_bridge
            .write_from_cols(
                io.p3.data.clone(),
                timestamp.clone() + timestamp_delta.clone(),
                &aux.write_p3_aux_cols.data,
            )
            .eval(builder, aux.aux.is_valid);
        timestamp_delta += AB::Expr::one();

        self.execution_bus.execute_increment_pc(
            builder,
            aux.aux.is_valid,
            io.from_state.map(Into::into),
            timestamp_delta,
        );
    }
}

impl EcDoubleVmAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: EcDoubleIoCols<AB::Var>,
        aux: EcDoubleAuxCols<AB::Var>,
    ) {
        let mut timestamp_delta = AB::Expr::zero();
        let timestamp: AB::Expr = io.from_state.timestamp.into();

        self.program_bus.send_instruction(
            builder,
            io.from_state.pc,
            AB::Expr::from_canonical_u8(Opcode::SECP256K1_EC_DOUBLE as u8),
            [
                io.p2.address.address.into(),
                io.p1.address.address.into(),
                AB::Expr::zero(),
                io.p1.address.address_space.into(),
                io.p1.data.address_space.into(),
            ],
            aux.aux.is_valid,
        );

        self.memory_bridge
            .read_from_cols(
                io.p1.address.clone(),
                timestamp.clone() + timestamp_delta.clone(),
                &aux.read_p1_aux_cols.address,
            )
            .eval(builder, aux.aux.is_valid);
        timestamp_delta += AB::Expr::one();
        self.memory_bridge
            .read_from_cols(
                io.p1.data.clone(),
                timestamp.clone() + timestamp_delta.clone(),
                &aux.read_p1_aux_cols.data,
            )
            .eval(builder, aux.aux.is_valid);
        timestamp_delta += AB::Expr::one();

        self.memory_bridge
            .read_from_cols(
                io.p2.address.clone(),
                timestamp.clone() + timestamp_delta.clone(),
                &aux.write_p2_aux_cols.address,
            )
            .eval(builder, aux.aux.is_valid);
        timestamp_delta += AB::Expr::one();
        self.memory_bridge
            .write_from_cols(
                io.p2.data.clone(),
                timestamp.clone() + timestamp_delta.clone(),
                &aux.write_p2_aux_cols.data,
            )
            .eval(builder, aux.aux.is_valid);
        timestamp_delta += AB::Expr::one();

        self.execution_bus.execute_increment_pc(
            builder,
            aux.aux.is_valid,
            io.from_state.map(Into::into),
            timestamp_delta,
        );
    }
}
