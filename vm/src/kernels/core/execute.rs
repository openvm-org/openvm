use std::{array, collections::BTreeMap};

use afs_primitives::{is_equal::IsEqualAir, sub_chip::LocalTraceInstructions};
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
    kernels::core::{
        columns::{CoreAuxCols, CoreCols, CoreIoCols, CoreMemoryAccessCols},
        CORE_MAX_READS_PER_CYCLE, CORE_MAX_WRITES_PER_CYCLE, INST_WIDTH,
    },
    system::{
        memory::offline_checker::{MemoryReadOrImmediateAuxCols, MemoryWriteAuxCols},
        program::{ExecutionError, Instruction},
    },
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

        let mut next_pc = pc + 1;

        let mut write_records = vec![];
        let mut read_records = vec![];

        macro_rules! read {
            ($addr_space: expr, $pointer: expr) => {{
                assert!(read_records.len() < CORE_MAX_READS_PER_CYCLE);
                read_records.push(
                    self.memory_controller
                        .borrow_mut()
                        .read_cell($addr_space, $pointer),
                );
                read_records[read_records.len() - 1].data[0]
            }};
        }

        macro_rules! write {
            ($addr_space: expr, $pointer: expr, $data: expr) => {{
                assert!(write_records.len() < CORE_MAX_WRITES_PER_CYCLE);
                write_records.push(self.memory_controller.borrow_mut().write_cell(
                    $addr_space,
                    $pointer,
                    $data,
                ));
            }};
        }

        let hint_stream = &mut self.streams.hint_stream;

        let local_opcode_index = CoreOpcode::from_usize(local_opcode_index);
        match local_opcode_index {
            // d[a] <- e[d[c] + b]
            LOADW => {
                let base_pointer = read!(d, c);
                let value = read!(e, base_pointer + b);
                write!(d, a, value);
            }
            // e[d[c] + b] <- d[a]
            STOREW => {
                let base_pointer = read!(d, c);
                let value = read!(d, a);
                write!(e, base_pointer + b, value);
            }
            // d[a] <- e[d[c] + b + d[f] * g]
            LOADW2 => {
                let base_pointer = read!(d, c);
                let index = read!(d, f);
                let value = read!(e, base_pointer + b + index * g);
                write!(d, a, value);
            }
            // e[d[c] + b + mem[f] * g] <- d[a]
            STOREW2 => {
                let base_pointer = read!(d, c);
                let value = read!(d, a);
                let index = read!(d, f);
                write!(e, base_pointer + b + index * g, value);
            }
            // d[a] <- pc + INST_WIDTH, pc <- pc + b
            JAL => {
                write!(d, a, F::from_canonical_u32(pc + INST_WIDTH));
                next_pc = (F::from_canonical_u32(pc) + b).as_canonical_u32();
            }
            // If d[a] = e[b], pc <- pc + c
            BEQ => {
                let left = read!(d, a);
                let right = read!(e, b);
                if left == right {
                    next_pc = (F::from_canonical_u32(pc) + c).as_canonical_u32();
                }
            }
            // If d[a] != e[b], pc <- pc + c
            BNE => {
                let left = read!(d, a);
                let right = read!(e, b);
                if left != right {
                    next_pc = (F::from_canonical_u32(pc) + c).as_canonical_u32();
                }
            }
            NOP => {
                unreachable!()
            }
            PRINTF => {
                let value = read!(d, a);
                println!("{}", value);
            }
            HINT_INPUT => {
                let hint = match self.streams.input_stream.pop_front() {
                    Some(hint) => hint,
                    None => {
                        return Err(ExecutionError::EndOfInputStream(pc));
                    }
                };
                hint_stream.clear();
                hint_stream.push_back(F::from_canonical_usize(hint.len()));
                hint_stream.extend(hint);
            }
            HINT_BITS => {
                let val = self.memory_controller.borrow().unsafe_read_cell(d, a);
                let mut val = val.as_canonical_u32();

                let len = c.as_canonical_u32();
                hint_stream.clear();
                for _ in 0..len {
                    hint_stream.push_back(F::from_canonical_u32(val & 1));
                    val >>= 1;
                }
            }
            HINT_BYTES => {
                let val = self.memory_controller.borrow().unsafe_read_cell(d, a);
                let mut val = val.as_canonical_u32();

                let len = c.as_canonical_u32();
                hint_stream.clear();
                for _ in 0..len {
                    hint_stream.push_back(F::from_canonical_u32(val & 0xff));
                    val >>= 8;
                }
            }
            // e[d[a] + b] <- hint_stream.next()
            SHINTW => {
                let hint = match hint_stream.pop_front() {
                    Some(hint) => hint,
                    None => {
                        return Err(ExecutionError::HintOutOfBounds(pc));
                    }
                };
                let base_pointer = read!(d, a);
                write!(e, base_pointer + b, hint);
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
            let aux_cols_factory = self.memory_controller.borrow().aux_cols_factory();

            let read_cols = array::from_fn(|i| {
                read_records
                    .get(i)
                    .map_or_else(CoreMemoryAccessCols::disabled, |read| {
                        CoreMemoryAccessCols::from_read_record(*read)
                    })
            });
            let reads_aux_cols = array::from_fn(|i| {
                read_records
                    .get(i)
                    .map_or_else(MemoryReadOrImmediateAuxCols::disabled, |read| {
                        aux_cols_factory.make_read_or_immediate_aux_cols(*read)
                    })
            });

            let write_cols = array::from_fn(|i| {
                write_records
                    .get(i)
                    .map_or_else(CoreMemoryAccessCols::disabled, |write| {
                        CoreMemoryAccessCols::from_write_record(*write)
                    })
            });
            let writes_aux_cols = array::from_fn(|i| {
                write_records
                    .get(i)
                    .map_or_else(MemoryWriteAuxCols::disabled, |write| {
                        aux_cols_factory.make_write_aux_cols(*write)
                    })
            });

            let mut operation_flags = BTreeMap::new();
            for other_opcode in CoreOpcode::iter() {
                operation_flags.insert(
                    other_opcode,
                    F::from_bool(other_opcode == local_opcode_index),
                );
            }

            let is_equal_cols = LocalTraceInstructions::generate_trace_row(
                &IsEqualAir,
                (read_cols[0].value, read_cols[1].value),
            );

            let read0_equals_read1 = is_equal_cols.io.is_equal;
            let is_equal_aux = is_equal_cols.aux;

            let aux = CoreAuxCols {
                operation_flags,
                reads: read_cols,
                writes: write_cols,
                read0_equals_read1,
                is_equal_aux,
                reads_aux_cols,
                writes_aux_cols,
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
