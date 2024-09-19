use afs_stark_backend::interaction::InteractionBuilder;
use itertools::izip;
use p3_field::AbstractField;

use super::{
    air::UintArithmeticAir,
    columns::{UintArithmeticAuxCols, UintArithmeticIoCols},
};
use crate::memory::MemoryAddress;

impl<const ARG_SIZE: usize, const LIMB_SIZE: usize> UintArithmeticAir<ARG_SIZE, LIMB_SIZE> {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: UintArithmeticIoCols<ARG_SIZE, LIMB_SIZE, AB::Var>,
        aux: UintArithmeticAuxCols<ARG_SIZE, LIMB_SIZE, AB::Var>,
        expected_opcode: AB::Expr,
    ) {
        let mut timestamp_delta = AB::Expr::zero();

        let timestamp: AB::Var = io.from_state.timestamp;

        // Interaction with program
        self.program_bus.send_instruction(
            builder,
            [
                io.from_state.pc.into(),
                expected_opcode.clone(),
                io.z.ptr.into(),
                io.x.ptr.into(),
                io.y.ptr.into(),
                io.d.into(),
                io.z.address_space.into(),
                io.x.address_space.into(),
                io.y.address_space.into(),
            ]
            .into_iter(),
            aux.is_valid,
        );

        // Read the operand pointer's values, which are themselves pointers
        // for the actual IO data.
        for (ptr, value, mem_aux) in izip!(
            [io.z.ptr, io.x.ptr, io.y.ptr],
            [io.z.address, io.x.address, io.y.address],
            &aux.read_ptr_aux_cols
        ) {
            self.memory_bridge
                .read(
                    MemoryAddress::new(io.d, ptr), // all use addr space d
                    [value],
                    timestamp + timestamp_delta.clone(),
                    mem_aux,
                )
                .eval(builder, aux.is_valid);

            timestamp_delta += AB::Expr::one();
        }

        // Memory read for x data
        self.memory_bridge
            .read(
                MemoryAddress::new(io.x.address_space, io.x.address),
                io.x.data.try_into().unwrap_or_else(|_| unreachable!()),
                timestamp + timestamp_delta.clone(),
                &aux.read_x_aux_cols,
            )
            .eval(builder, aux.is_valid);
        timestamp_delta += AB::Expr::one();

        // Memory read for y data
        self.memory_bridge
            .read(
                MemoryAddress::new(io.y.address_space, io.y.address),
                io.y.data.try_into().unwrap_or_else(|_| unreachable!()),
                timestamp + timestamp_delta.clone(),
                &aux.read_y_aux_cols,
            )
            .eval(builder, aux.is_valid);
        timestamp_delta += AB::Expr::one();

        // Special handling for writing output z data:
        let enabled = aux.opcode_add_flag + aux.opcode_sub_flag;
        self.memory_bridge
            .write(
                MemoryAddress::new(io.z.address_space, io.z.address),
                io.z.data
                    .clone()
                    .try_into()
                    .unwrap_or_else(|_| unreachable!()),
                timestamp + timestamp_delta.clone(),
                &aux.write_z_aux_cols,
            )
            .eval(builder, enabled.clone());
        timestamp_delta += enabled;

        let enabled = aux.opcode_lt_flag + aux.opcode_eq_flag;
        self.memory_bridge
            .write(
                MemoryAddress::new(io.z.address_space, io.z.address),
                [io.cmp_result],
                timestamp + timestamp_delta.clone(),
                &aux.write_cmp_aux_cols,
            )
            .eval(builder, enabled.clone());
        timestamp_delta += enabled;

        self.execution_bus.execute_increment_pc(
            builder,
            aux.is_valid,
            io.from_state.map(Into::into),
            timestamp_delta,
        );

        // Chip-specific interactions
        for z in io.z.data.iter() {
            self.bus.range_check(*z, LIMB_SIZE).eval(
                builder,
                aux.opcode_add_flag + aux.opcode_sub_flag + aux.opcode_lt_flag,
            );
        }
    }
}
