use std::collections::BTreeMap;

use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::{columns::CpuIoCols, CpuAir, READ_INSTRUCTION_BUS};
use crate::arch::{
    columns::{ExecutionState, InstructionCols},
    instructions::Opcode,
};

impl<const WORD_SIZE: usize> CpuAir<WORD_SIZE> {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: CpuIoCols<AB::Var>,
        next_io: CpuIoCols<AB::Var>,
        operation_flags: &BTreeMap<Opcode, AB::Var>,
        not_cpu_opcode: AB::Expr,
    ) {
        // Interaction with program (bus 0)
        builder.push_send(
            READ_INSTRUCTION_BUS,
            [
                io.pc, io.opcode, io.op_a, io.op_b, io.op_c, io.d, io.e, io.op_f, io.op_g,
            ],
            AB::Expr::one() - operation_flags[&Opcode::NOP],
        );

        self.execution_bus.execute(
            builder,
            -not_cpu_opcode,
            ExecutionState::new(io.pc, io.timestamp),
            ExecutionState::new(next_io.pc, next_io.timestamp),
            InstructionCols::new(
                io.opcode,
                [io.op_a, io.op_b, io.op_c, io.d, io.e, io.op_f, io.op_g],
            ),
        );

        /*if self.options.is_less_than_enabled {
            let fields = [
                accesses[0].data[0],
                accesses[1].data[0],
                accesses[CPU_MAX_READS_PER_CYCLE].data[0],
            ];
            let count = operation_flags[&F_LESS_THAN];
            builder.push_send(IS_LESS_THAN_BUS, fields, count);
        }*/
    }
}
