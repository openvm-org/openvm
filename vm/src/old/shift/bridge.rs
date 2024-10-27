use ax_stark_backend::interaction::InteractionBuilder;
use itertools::izip;
use p3_field::AbstractField;

use super::{
    air::ShiftCoreAir,
    columns::{ShiftAuxCols, ShiftIoCols},
};
use crate::system::memory::MemoryAddress;

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize> ShiftCoreAir<NUM_LIMBS, LIMB_BITS> {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: &ShiftIoCols<AB::Var, NUM_LIMBS, LIMB_BITS>,
        aux: &ShiftAuxCols<AB::Var, NUM_LIMBS, LIMB_BITS>,
        expected_opcode: AB::Expr,
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
                expected_opcode + AB::Expr::from_canonical_usize(self.offset),
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

        // Check that bit_shift < LIMB_BITS
        self.range_bus
            .range_check(aux.bit_shift, LIMB_BITS.ilog2() as usize)
            .eval(builder, aux.is_valid);

        // Check x_sign & x[NUM_LIMBS - 1] == x_sign using XOR
        let mask = AB::F::from_canonical_u32(1 << (LIMB_BITS - 1));
        let x_sign_shifted = aux.x_sign * mask;
        self.xor_bus
            .send(
                io.x.data[NUM_LIMBS - 1],
                mask,
                io.x.data[NUM_LIMBS - 1] + mask
                    - (AB::Expr::from_canonical_u32(2) * x_sign_shifted),
            )
            .eval(builder, aux.opcode_sra_flag);

        for (z, carry) in io.z.data.iter().zip(aux.bit_shift_carry.iter()) {
            self.range_bus
                .range_check(*z, LIMB_BITS)
                .eval(builder, aux.is_valid);
            self.range_bus
                .send(*carry, aux.bit_shift)
                .eval(builder, aux.is_valid);
        }
    }
}
