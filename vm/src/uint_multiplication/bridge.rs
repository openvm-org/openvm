use afs_stark_backend::interaction::InteractionBuilder;
use itertools::izip;
use p3_field::AbstractField;

use super::{
    air::UintMultiplicationAir,
    columns::{UintMultiplicationAuxCols, UintMultiplicationIoCols},
};
use crate::{
    arch::{columns::InstructionCols, instructions::Opcode},
    memory::{offline_checker::MemoryBridge, MemoryAddress},
};

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize> UintMultiplicationAir<NUM_LIMBS, LIMB_BITS> {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: UintMultiplicationIoCols<NUM_LIMBS, LIMB_BITS, AB::Var>,
        aux: UintMultiplicationAuxCols<NUM_LIMBS, LIMB_BITS, AB::Var>,
    ) {
        let memory_bridge = MemoryBridge::new(self.mem_oc);
        let timestamp: AB::Var = io.from_state.timestamp;
        let mut timestamp_delta = AB::Expr::zero();

        for (ptr, value, mem_aux) in izip!(
            [
                io.z.ptr_to_address,
                io.x.ptr_to_address,
                io.y.ptr_to_address
            ],
            [io.z.address, io.x.address, io.y.address],
            aux.read_ptr_aux_cols
        ) {
            memory_bridge
                .read(
                    MemoryAddress::new(io.d, ptr),
                    [value],
                    timestamp + timestamp_delta.clone(),
                    &mem_aux,
                )
                .eval(builder, aux.is_valid);
            timestamp_delta += AB::Expr::one();
        }

        memory_bridge
            .read(
                MemoryAddress::new(io.e, io.x.address),
                io.x.data.try_into().unwrap_or_else(|_| unreachable!()),
                timestamp + timestamp_delta.clone(),
                &aux.read_x_aux_cols,
            )
            .eval(builder, aux.is_valid);
        timestamp_delta += AB::Expr::one();

        memory_bridge
            .read(
                MemoryAddress::new(io.e, io.y.address),
                io.y.data.try_into().unwrap_or_else(|_| unreachable!()),
                timestamp + timestamp_delta.clone(),
                &aux.read_y_aux_cols,
            )
            .eval(builder, aux.is_valid);
        timestamp_delta += AB::Expr::one();

        memory_bridge
            .write(
                MemoryAddress::new(io.e, io.z.address),
                io.z.data
                    .clone()
                    .try_into()
                    .unwrap_or_else(|_| unreachable!()),
                timestamp + timestamp_delta.clone(),
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
                AB::Expr::from_canonical_u8(Opcode::MUL256 as u8),
                [
                    io.z.ptr_to_address,
                    io.x.ptr_to_address,
                    io.y.ptr_to_address,
                    io.d,
                    io.e,
                ],
            ),
        );

        for (z, carry) in io.z.data.iter().zip(aux.carry.iter()) {
            self.bus.send(vec![*z, *carry]).eval(builder, aux.is_valid);
        }
    }
}
