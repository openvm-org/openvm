use afs_stark_backend::interaction::InteractionBuilder;
use itertools::izip;
use p3_field::AbstractField;

use super::{
    air::UintMultiplicationAir,
    columns::{UintMultiplicationAuxCols, UintMultiplicationIoCols},
};
use crate::{arch::instructions::Opcode, memory::MemoryAddress};

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize> UintMultiplicationAir<NUM_LIMBS, LIMB_BITS> {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: &UintMultiplicationIoCols<AB::Var, NUM_LIMBS, LIMB_BITS>,
        aux: &UintMultiplicationAuxCols<AB::Var, NUM_LIMBS, LIMB_BITS>,
    ) {
        let timestamp: AB::Var = io.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        for (ptr, value, mem_aux) in izip!(
            [
                io.z.ptr_to_address,
                io.x.ptr_to_address,
                io.y.ptr_to_address
            ],
            [io.z.address, io.x.address, io.y.address],
            &aux.read_ptr_aux_cols
        ) {
            self.memory_bridge
                .read(
                    MemoryAddress::new(io.ptr_as, ptr),
                    [value],
                    timestamp_pp(),
                    mem_aux,
                )
                .eval(builder, aux.is_valid);
        }

        self.memory_bridge
            .read(
                MemoryAddress::new(io.address_as, io.x.address),
                io.x.data,
                timestamp_pp(),
                &aux.read_x_aux_cols,
            )
            .eval(builder, aux.is_valid);

        self.memory_bridge
            .read(
                MemoryAddress::new(io.address_as, io.y.address),
                io.y.data,
                timestamp_pp(),
                &aux.read_y_aux_cols,
            )
            .eval(builder, aux.is_valid);

        self.memory_bridge
            .write(
                MemoryAddress::new(io.address_as, io.z.address),
                io.z.data,
                timestamp_pp(),
                &aux.write_z_aux_cols,
            )
            .eval(builder, aux.is_valid);

        self.execution_bridge
            .execute_and_increment_pc(
                AB::Expr::from_canonical_u8(Opcode::MUL256 as u8),
                [
                    io.z.ptr_to_address,
                    io.x.ptr_to_address,
                    io.y.ptr_to_address,
                    io.ptr_as,
                    io.address_as,
                ],
                io.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
            )
            .eval(builder, aux.is_valid);

        for (z, carry) in io.z.data.iter().zip(aux.carry.iter()) {
            self.bus.send(vec![*z, *carry]).eval(builder, aux.is_valid);
        }
    }
}
