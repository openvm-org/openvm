use afs_stark_backend::interaction::InteractionBuilder;
use itertools::izip;
use p3_field::AbstractField;

use super::{
    air::UintArithmeticAir,
    columns::{UintArithmeticAuxCols, UintArithmeticIoCols},
};
use crate::memory::MemoryAddress;

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize> UintArithmeticAir<NUM_LIMBS, LIMB_BITS> {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: &UintArithmeticIoCols<AB::Var, NUM_LIMBS, LIMB_BITS>,
        aux: &UintArithmeticAuxCols<AB::Var, NUM_LIMBS, LIMB_BITS>,
        expected_opcode: AB::Expr,
    ) {
        let timestamp: AB::Var = io.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        let range_check =
            aux.opcode_add_flag + aux.opcode_sub_flag + aux.opcode_lt_flag + aux.opcode_slt_flag;
        let bitwise = aux.opcode_xor_flag + aux.opcode_and_flag + aux.opcode_or_flag;

        // Read the operand pointer's values, which are themselves pointers
        // for the actual IO data.
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

        // Special handling for writing output z data:
        let enabled = aux.opcode_add_flag + aux.opcode_sub_flag + bitwise.clone();
        self.memory_bridge
            .write(
                MemoryAddress::new(io.address_as, io.z.address),
                io.z.data,
                timestamp + AB::F::from_canonical_usize(timestamp_delta),
                &aux.write_z_aux_cols,
            )
            .eval(builder, enabled);

        let enabled = aux.opcode_lt_flag + aux.opcode_eq_flag + aux.opcode_slt_flag;
        self.memory_bridge
            .write(
                MemoryAddress::new(io.address_as, io.z.address),
                [io.cmp_result],
                timestamp + AB::F::from_canonical_usize(timestamp_delta),
                &aux.write_cmp_aux_cols,
            )
            .eval(builder, enabled.clone());
        timestamp_delta += 1;

        self.execution_bridge
            .execute_and_increment_pc(
                expected_opcode,
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

        // Chip-specific interactions
        for i in 0..NUM_LIMBS {
            let x = range_check.clone() * io.z.data[i] + bitwise.clone() * io.x.data[i];
            let y = range_check.clone() * io.z.data[i] + bitwise.clone() * io.y.data[i];
            let x_or_y = aux.opcode_xor_flag * io.z.data[i]
                + aux.opcode_and_flag
                    * (io.x.data[i] + io.y.data[i]
                        - (AB::Expr::from_canonical_u32(2) * io.z.data[i]))
                + aux.opcode_or_flag
                    * ((AB::Expr::from_canonical_u32(2) * io.z.data[i])
                        - io.x.data[i]
                        - io.y.data[i]);
            self.bus
                .send(x, y, x_or_y)
                .eval(builder, range_check.clone() + bitwise.clone());
        }
    }
}
