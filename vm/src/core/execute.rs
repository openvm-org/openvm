use std::{array, collections::BTreeMap};

use afs_primitives::{is_equal::IsEqualAir, sub_chip::LocalTraceInstructions};
use p3_field::PrimeField32;
use strum::IntoEnumIterator;

use super::{timestamp_delta, CoreChip, CoreState};
use crate::{
    arch::{
        instructions::{
            CoreOpcode::{self, *},
            UsizeOpcode,
        },
        ExecutionState, InstructionExecutor,
    },
    core::{
        columns::{CoreAuxCols, CoreCols, CoreIoCols, CoreMemoryAccessCols},
        CORE_MAX_READS_PER_CYCLE, CORE_MAX_WRITES_PER_CYCLE, INST_WIDTH,
    },
    memory::offline_checker::{MemoryReadOrImmediateAuxCols, MemoryWriteAuxCols},
    program::{ExecutionError, Instruction},
};

impl<F: PrimeField32> InstructionExecutor<F> for CoreChip<F> {
    fn execute(
        &mut self,
        instruction: Instruction<F>,
        from_state: ExecutionState<usize>,
    ) -> Result<ExecutionState<usize>, ExecutionError> {
        let mut timestamp = from_state.timestamp;
        let pc = F::from_canonical_usize(from_state.pc);

        let core_options = self.air.options;
        let num_public_values = core_options.num_public_values;

        let pc_usize = pc.as_canonical_u64() as usize;

        let opcode = instruction.opcode - self.offset;
        let a = instruction.op_a;
        let b = instruction.op_b;
        let c = instruction.op_c;
        let d = instruction.d;
        let e = instruction.e;
        let f = instruction.op_f;
        let g = instruction.op_g;

        let io = CoreIoCols {
            timestamp: F::from_canonical_usize(timestamp),
            pc,
            opcode: F::from_canonical_usize(opcode),
            op_a: a,
            op_b: b,
            op_c: c,
            d,
            e,
            op_f: f,
            op_g: g,
        };

        let mut next_pc = pc + F::one();

        let mut write_records = vec![];
        let mut read_records = vec![];

        macro_rules! read {
            ($addr_space: expr, $pointer: expr) => {{
                assert!(read_records.len() < CORE_MAX_READS_PER_CYCLE);
                read_records.push(
                    self.memory_chip
                        .borrow_mut()
                        .read_cell($addr_space, $pointer),
                );
                read_records[read_records.len() - 1].data[0]
            }};
        }

        macro_rules! write {
            ($addr_space: expr, $pointer: expr, $data: expr) => {{
                assert!(write_records.len() < CORE_MAX_WRITES_PER_CYCLE);
                write_records.push(self.memory_chip.borrow_mut().write_cell(
                    $addr_space,
                    $pointer,
                    $data,
                ));
            }};
        }

        let mut public_value_flags = vec![F::zero(); num_public_values];

        let hint_stream = &mut self.streams.hint_stream;

        let opcode = CoreOpcode::from_usize(opcode);
        match opcode {
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
                write!(d, a, pc + F::from_canonical_usize(INST_WIDTH));
                next_pc = pc + b;
            }
            // If d[a] = e[b], pc <- pc + c
            BEQ => {
                let left = read!(d, a);
                let right = read!(e, b);
                if left == right {
                    next_pc = pc + c;
                }
            }
            // If d[a] != e[b], pc <- pc + c
            BNE => {
                let left = read!(d, a);
                let right = read!(e, b);
                if left != right {
                    next_pc = pc + c;
                }
            }
            TERMINATE | NOP => {
                next_pc = pc;
            }
            PUBLISH => {
                let public_value_index = read!(d, a).as_canonical_u64() as usize;
                let value = read!(e, b);
                if public_value_index >= num_public_values {
                    return Err(ExecutionError::PublicValueIndexOutOfBounds(
                        pc_usize,
                        num_public_values,
                        public_value_index,
                    ));
                }
                public_value_flags[public_value_index] = F::one();

                let public_values = &mut self.public_values;
                match public_values[public_value_index] {
                    None => public_values[public_value_index] = Some(value),
                    Some(exising_value) => {
                        if value != exising_value {
                            return Err(ExecutionError::PublicValueNotEqual(
                                pc_usize,
                                public_value_index,
                                exising_value.as_canonical_u64() as usize,
                                value.as_canonical_u64() as usize,
                            ));
                        }
                    }
                }
            }
            PRINTF => {
                let value = read!(d, a);
                println!("{}", value);
            }
            HINT_INPUT => {
                let hint = match self.streams.input_stream.pop_front() {
                    Some(hint) => hint,
                    None => {
                        return Err(ExecutionError::EndOfInputStream(pc_usize));
                    }
                };
                hint_stream.clear();
                hint_stream.push_back(F::from_canonical_usize(hint.len()));
                hint_stream.extend(hint);
            }
            HINT_BITS => {
                let val = self.memory_chip.borrow().unsafe_read_cell(d, a);
                let mut val = val.as_canonical_u32();

                let len = c.as_canonical_u32();
                hint_stream.clear();
                for _ in 0..len {
                    hint_stream.push_back(F::from_canonical_u32(val & 1));
                    val >>= 1;
                }
            }
            HINT_BYTES => {
                let val = self.memory_chip.borrow().unsafe_read_cell(d, a);
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
                        return Err(ExecutionError::HintOutOfBounds(pc_usize));
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
        timestamp += timestamp_delta(opcode);

        // TODO[zach]: Only collect a record of { from_state, instruction, read_records, write_records, public_value_index }
        // and move this logic into generate_trace().
        {
            let aux_cols_factory = self.memory_chip.borrow().aux_cols_factory();

            let read_cols = array::from_fn(|i| {
                read_records
                    .get(i)
                    .map_or_else(CoreMemoryAccessCols::disabled, |read| {
                        CoreMemoryAccessCols::from_read_record(read.clone())
                    })
            });
            let reads_aux_cols = array::from_fn(|i| {
                read_records
                    .get(i)
                    .map_or_else(MemoryReadOrImmediateAuxCols::disabled, |read| {
                        aux_cols_factory.make_read_or_immediate_aux_cols(read.clone())
                    })
            });

            let write_cols = array::from_fn(|i| {
                write_records
                    .get(i)
                    .map_or_else(CoreMemoryAccessCols::disabled, |write| {
                        CoreMemoryAccessCols::from_write_record(write.clone())
                    })
            });
            let writes_aux_cols = array::from_fn(|i| {
                write_records
                    .get(i)
                    .map_or_else(MemoryWriteAuxCols::disabled, |write| {
                        aux_cols_factory.make_write_aux_cols(write.clone())
                    })
            });

            let mut operation_flags = BTreeMap::new();
            for other_opcode in CoreOpcode::iter() {
                operation_flags.insert(other_opcode, F::from_bool(other_opcode == opcode));
            }

            let is_equal_cols = LocalTraceInstructions::generate_trace_row(
                &IsEqualAir,
                (read_cols[0].value, read_cols[1].value),
            );

            let read0_equals_read1 = is_equal_cols.io.is_equal;
            let is_equal_aux = is_equal_cols.aux;

            let aux = CoreAuxCols {
                operation_flags,
                public_value_flags,
                reads: read_cols,
                writes: write_cols,
                read0_equals_read1,
                is_equal_aux,
                reads_aux_cols,
                writes_aux_cols,
                next_pc,
            };

            let cols = CoreCols { io, aux };
            self.rows.push(cols.flatten());
        }

        // Update Core chip state with all changes from this segment.
        self.set_state(CoreState {
            clock_cycle: self.state.clock_cycle + 1,
            timestamp,
            pc: next_pc.as_canonical_u64() as usize,
            is_done: opcode == TERMINATE,
        });

        Ok(ExecutionState::new(
            next_pc.as_canonical_u64() as usize,
            timestamp,
        ))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        let opcode = CoreOpcode::from_usize(opcode - self.offset);
        format!("{opcode:?}")
    }
}
