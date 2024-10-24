use std::collections::BTreeMap;

use p3_field::PrimeField32;
use strum::IntoEnumIterator;

use super::CoreChip;
use crate::{
    arch::{
        instructions::{
            CoreOpcode::{self, *},
            UsizeOpcode,
        },
        ExecutionState, InstructionExecutor,
    },
    kernels::core::columns::{CoreAuxCols, CoreCols, CoreIoCols},
    system::program::{ExecutionError, Instruction},
};

impl<F: PrimeField32> InstructionExecutor<F> for CoreChip<F> {
    fn execute(
        &mut self,
        instruction: Instruction<F>,
        from_state: ExecutionState<u32>,
    ) -> Result<ExecutionState<u32>, ExecutionError> {
        let ExecutionState { pc, timestamp } = from_state;

        let local_opcode_index = instruction.opcode - self.offset;
        let Instruction { a, b, c, d, e, .. } = instruction;

        let io = CoreIoCols {
            timestamp: F::from_canonical_u32(timestamp),
            pc: F::from_canonical_u32(pc),
            opcode: F::from_canonical_usize(local_opcode_index),
            a,
            b,
            c,
            d,
            e,
        };

        let mut streams = self.streams.lock();

        let local_opcode_index = CoreOpcode::from_usize(local_opcode_index);
        match local_opcode_index {
            PRINTF => {
                let value = self.memory_controller.borrow().unsafe_read_cell(d, a);
                println!("{}", value);
            }
            HINT_INPUT => {
                let hint = match streams.input_stream.pop_front() {
                    Some(hint) => hint,
                    None => {
                        return Err(ExecutionError::EndOfInputStream(pc));
                    }
                };
                streams.hint_stream.clear();
                streams
                    .hint_stream
                    .push_back(F::from_canonical_usize(hint.len()));
                streams.hint_stream.extend(hint);
            }
            HINT_BITS => {
                let val = self.memory_controller.borrow().unsafe_read_cell(d, a);
                let mut val = val.as_canonical_u32();

                let len = c.as_canonical_u32();
                streams.hint_stream.clear();
                for _ in 0..len {
                    streams
                        .hint_stream
                        .push_back(F::from_canonical_u32(val & 1));
                    val >>= 1;
                }
            }
            HINT_BYTES => {
                let val = self.memory_controller.borrow().unsafe_read_cell(d, a);
                let mut val = val.as_canonical_u32();

                let len = c.as_canonical_u32();
                streams.hint_stream.clear();
                for _ in 0..len {
                    streams
                        .hint_stream
                        .push_back(F::from_canonical_u32(val & 0xff));
                    val >>= 8;
                }
            }
            CT_START | CT_END => {
                // Advance program counter, but don't do anything else
                // TODO: move handling of these instructions outside CoreChip
            }
            _ => unreachable!(),
        };

        // TODO[zach]: Only collect a record of { from_state, instruction, read_records, write_records }
        // and move this logic into generate_trace().
        {
            let mut operation_flags = BTreeMap::new();
            for other_opcode in CoreOpcode::iter() {
                operation_flags.insert(
                    other_opcode,
                    F::from_bool(other_opcode == local_opcode_index),
                );
            }

            let aux = CoreAuxCols {
                is_valid: F::one(),
                operation_flags,
            };

            let cols = CoreCols { io, aux };
            self.rows.push(cols.flatten());
        }

        self.memory_controller.borrow_mut().increment_timestamp();
        Ok(ExecutionState::new(
            pc + 1,
            self.memory_controller.borrow().timestamp(),
        ))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        let local_opcode_index = CoreOpcode::from_usize(opcode - self.offset);
        format!("{local_opcode_index:?}")
    }
}
