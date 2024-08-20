use std::collections::BTreeMap;

use p3_field::AbstractField;

use afs_stark_backend::interaction::InteractionBuilder;

use crate::arch::columns::{ExecutionState, InstructionCols};

use super::{
    columns::{CpuIoCols, MemoryAccessCols},
    CPU_MAX_ACCESSES_PER_CYCLE, CPU_MAX_READS_PER_CYCLE, CpuAir, MEMORY_BUS
    , OpCode, READ_INSTRUCTION_BUS,
};

impl<const WORD_SIZE: usize> CpuAir<WORD_SIZE> {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: CpuIoCols<AB::Var>,
        next_io: CpuIoCols<AB::Var>,
        accesses: [MemoryAccessCols<WORD_SIZE, AB::Var>; CPU_MAX_ACCESSES_PER_CYCLE],
        operation_flags: &BTreeMap<OpCode, AB::Var>,
        not_cpu_opcode: AB::Expr,
    ) {
        // Interaction with program (bus 0)
        builder.push_send(
            READ_INSTRUCTION_BUS,
            [
                io.pc, io.opcode, io.op_a, io.op_b, io.op_c, io.d, io.e, io.op_f, io.op_g,
            ],
            AB::Expr::one() - operation_flags[&OpCode::NOP],
        );

        for (i, access) in accesses.into_iter().enumerate() {
            let memory_cycle = io.timestamp + AB::F::from_canonical_usize(i);
            let is_write = i >= CPU_MAX_READS_PER_CYCLE;

            let fields = [
                memory_cycle,
                AB::F::from_bool(is_write).into(),
                access.address_space.into(),
                access.address.into(),
            ]
            .into_iter()
            .chain(access.data.into_iter().map(Into::into));
            builder.push_send(MEMORY_BUS, fields, access.enabled - access.is_immediate);
        }

        self.execution_bus.execute(
            builder,
            not_cpu_opcode,
            ExecutionState::new(io.pc, io.timestamp),
            ExecutionState::new(next_io.pc, next_io.timestamp),
            InstructionCols::new_large(
                io.opcode, io.op_a, io.op_b, io.op_c, io.d, io.e, io.op_f, io.op_g,
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
