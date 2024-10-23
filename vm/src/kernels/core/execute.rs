use std::collections::BTreeMap;

use p3_field::PrimeField32;
use strum::IntoEnumIterator;

use super::{timestamp_delta, CoreChip};
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
        let ExecutionState { pc, mut timestamp } = from_state;

        let local_opcode_index = instruction.opcode - self.offset;
        let a = instruction.a;
        let b = instruction.b;
        let c = instruction.c;
        let d = instruction.d;
        let e = instruction.e;
        let f = instruction.f;
        let g = instruction.g;

        let io = CoreIoCols {
            timestamp: F::from_canonical_u32(timestamp),
            pc: F::from_canonical_u32(pc),
            opcode: F::from_canonical_usize(local_opcode_index),
            a,
            b,
            c,
            d,
            e,
            f,
            g,
        };

        let next_pc = pc + 1;

        macro_rules! read {
            ($addr_space: expr, $pointer: expr) => {{
                self.memory_controller
                    .borrow_mut()
                    .read_cell($addr_space, $pointer)
                    .data[0]
            }};
        }

        let mut streams = self.streams.lock();

        let local_opcode_index = CoreOpcode::from_usize(local_opcode_index);
        match local_opcode_index {
            DUMMY => {
                unreachable!()
            }
            PRINTF => {
                let value = read!(d, a);
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
        timestamp += timestamp_delta(local_opcode_index);

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
                operation_flags,
                next_pc: F::from_canonical_u32(next_pc),
            };

            let cols = CoreCols { io, aux };
            self.rows.push(cols.flatten());
        }

        Ok(ExecutionState::new(next_pc, timestamp))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        let local_opcode_index = CoreOpcode::from_usize(opcode - self.offset);
        format!("{local_opcode_index:?}")
    }
}
