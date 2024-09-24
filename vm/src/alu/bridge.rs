use afs_stark_backend::interaction::InteractionBuilder;
use itertools::izip;
use p3_field::AbstractField;

use super::{
    air::ArithmeticLogicAir,
    columns::{ArithmeticLogicAuxCols, ArithmeticLogicIoCols},
};
use crate::memory::MemoryAddress;

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize> ArithmeticLogicAir<NUM_LIMBS, LIMB_BITS> {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: &ArithmeticLogicIoCols<AB::Var, NUM_LIMBS, LIMB_BITS>,
        aux: &ArithmeticLogicAuxCols<AB::Var, NUM_LIMBS, LIMB_BITS>,
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
        self.memory_bridge
            .write(
                MemoryAddress::new(io.address_as, io.z.address),
                io.z.data,
                timestamp + AB::F::from_canonical_usize(timestamp_delta),
                &aux.write_z_aux_cols,
            )
            .eval(
                builder,
                aux.opcode_add_flag + aux.opcode_sub_flag + bitwise.clone(),
            );

        // Special handling for writing output cmp data:
        self.memory_bridge
            .write(
                MemoryAddress::new(io.address_as, io.z.address),
                [io.cmp_result],
                timestamp + AB::F::from_canonical_usize(timestamp_delta),
                &aux.write_cmp_aux_cols,
            )
            .eval(
                builder,
                aux.opcode_lt_flag + aux.opcode_eq_flag + aux.opcode_slt_flag,
            );
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

        // Check x_sign & x[NUM_LIMBS - 1] == x_sign using XOR
        let x_sign_shifted = aux.x_sign * AB::F::from_canonical_u32(1 << (LIMB_BITS - 1));
        let y_sign_shifted = aux.y_sign * AB::F::from_canonical_u32(1 << (LIMB_BITS - 1));
        self.bus
            .send(
                x_sign_shifted.clone(),
                io.x.data[NUM_LIMBS - 1],
                io.x.data[NUM_LIMBS - 1] - x_sign_shifted,
            )
            .eval(builder, aux.opcode_slt_flag);
        self.bus
            .send(
                y_sign_shifted.clone(),
                io.y.data[NUM_LIMBS - 1],
                io.y.data[NUM_LIMBS - 1] - y_sign_shifted,
            )
            .eval(builder, aux.opcode_slt_flag);

        // Chip-specific interactions
        for i in 0..NUM_LIMBS {
            let x = range_check.clone() * io.z.data[i] + bitwise.clone() * io.x.data[i];
            let y = range_check.clone() * io.z.data[i] + bitwise.clone() * io.y.data[i];
            let xor_res = aux.opcode_xor_flag * io.z.data[i]
                + aux.opcode_and_flag
                    * (io.x.data[i] + io.y.data[i]
                        - (AB::Expr::from_canonical_u32(2) * io.z.data[i]))
                + aux.opcode_or_flag
                    * ((AB::Expr::from_canonical_u32(2) * io.z.data[i])
                        - io.x.data[i]
                        - io.y.data[i]);
            self.bus
                .send(x, y, xor_res)
                .eval(builder, range_check.clone() + bitwise.clone());
        }
    }
}
