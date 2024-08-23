use std::collections::BTreeMap;

use p3_field::AbstractField;

use afs_stark_backend::interaction::InteractionBuilder;

use crate::{
    cpu::{
        IS_LESS_THAN_BUS,
        Opcode::{COMP_POS2, F_LESS_THAN, PERM_POS2},
    },
    memory::manager::operation::MemoryOperation,
};

use super::{
    ARITHMETIC_BUS, columns::CpuIoCols, CPU_MAX_ACCESSES_PER_CYCLE, CPU_MAX_READS_PER_CYCLE, CpuAir,
    FIELD_ARITHMETIC_INSTRUCTIONS, FIELD_EXTENSION_BUS, FIELD_EXTENSION_INSTRUCTIONS,
    Opcode, POSEIDON2_BUS, READ_INSTRUCTION_BUS,
};

impl<const WORD_SIZE: usize> CpuAir<WORD_SIZE> {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: CpuIoCols<AB::Var>,
        ops: [MemoryOperation<WORD_SIZE, AB::Var>; CPU_MAX_ACCESSES_PER_CYCLE],
        operation_flags: &BTreeMap<Opcode, AB::Var>,
    ) {
        // Interaction with program (bus 0)
        builder.push_send(
            READ_INSTRUCTION_BUS,
            [
                io.pc, io.opcode, io.op_a, io.op_b, io.op_c, io.d, io.e, io.op_f, io.op_g,
            ],
            AB::Expr::one() - operation_flags[&Opcode::NOP],
        );

        // Interaction with arithmetic (bus 2)
        if self.options.field_arithmetic_enabled {
            let fields = [
                io.opcode,
                ops[0].cell.data[0],
                ops[1].cell.data[0],
                ops[CPU_MAX_READS_PER_CYCLE].cell.data[0],
            ];
            let count = FIELD_ARITHMETIC_INSTRUCTIONS
                .iter()
                .fold(AB::Expr::zero(), |acc, opcode| {
                    acc + operation_flags[opcode]
                });
            builder.push_send(ARITHMETIC_BUS, fields, count);
        }

        // Interaction with field extension arithmetic (bus 3)
        if self.options.field_extension_enabled {
            let fields = [
                io.opcode,
                io.timestamp,
                io.op_a,
                io.op_b,
                io.op_c,
                io.d,
                io.e,
            ];
            let count = FIELD_EXTENSION_INSTRUCTIONS
                .iter()
                .fold(AB::Expr::zero(), |acc, opcode| {
                    acc + operation_flags[opcode]
                });
            builder.push_send(FIELD_EXTENSION_BUS, fields, count);
        }

        // Interaction with poseidon2 (bus 5)
        if self.options.poseidon2_enabled() {
            let compression = io.opcode - AB::F::from_canonical_usize(PERM_POS2 as usize);
            let fields = [io.timestamp, io.op_a, io.op_b, io.op_c, io.d, io.e]
                .into_iter()
                .map(Into::into)
                .chain([compression]);

            let mut count = AB::Expr::zero();
            if self.options.compress_poseidon2_enabled {
                count = count + operation_flags[&COMP_POS2];
            }
            if self.options.perm_poseidon2_enabled {
                count = count + operation_flags[&PERM_POS2];
            }
            builder.push_send(POSEIDON2_BUS, fields, count);
        }

        if self.options.is_less_than_enabled {
            let fields = [
                ops[0].cell.data[0],
                ops[1].cell.data[0],
                ops[CPU_MAX_READS_PER_CYCLE].cell.data[0],
            ];
            let count = operation_flags[&F_LESS_THAN];
            builder.push_send(IS_LESS_THAN_BUS, fields, count);
        }
    }
}
