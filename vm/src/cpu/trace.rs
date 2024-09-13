use std::{array, collections::BTreeMap};

use afs_primitives::{is_equal_vec::IsEqualVecAir, sub_chip::LocalTraceInstructions};
use afs_stark_backend::rap::AnyRap;
use p3_air::BaseAir;
use p3_commit::PolynomialSpace;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig};

use super::{
    columns::{CpuAuxCols, CpuCols, CpuIoCols},
    timestamp_delta, CpuChip, CpuState, CPU_MAX_READS_PER_CYCLE, CPU_MAX_WRITES_PER_CYCLE,
    INST_WIDTH,
};
use crate::{
    arch::{
        chips::{InstructionExecutor, MachineChip},
        columns::ExecutionState,
        instructions::{Opcode::*, CORE_INSTRUCTIONS},
    },
    cpu::{columns::CpuMemoryAccessCols, WORD_SIZE},
    memory::offline_checker::{MemoryReadOrImmediateAuxCols, MemoryWriteAuxCols},
    program::{ExecutionError, Instruction},
};

impl<F: PrimeField32> InstructionExecutor<F> for CpuChip<F> {
    fn execute(
        &mut self,
        instruction: Instruction<F>,
        from_state: ExecutionState<usize>,
    ) -> Result<ExecutionState<usize>, ExecutionError> {
        let mut timestamp = from_state.timestamp;
        let pc = F::from_canonical_usize(from_state.pc);

        let cpu_options = self.air.options;
        let num_public_values = cpu_options.num_public_values;

        let pc_usize = pc.as_canonical_u64() as usize;

        let opcode = instruction.opcode;
        let a = instruction.op_a;
        let b = instruction.op_b;
        let c = instruction.op_c;
        let d = instruction.d;
        let e = instruction.e;
        let f = instruction.op_f;
        let g = instruction.op_g;
        let debug = instruction.debug.clone();

        let io = CpuIoCols {
            timestamp: F::from_canonical_usize(timestamp),
            pc,
            opcode: F::from_canonical_usize(opcode as usize),
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
                assert!(read_records.len() < CPU_MAX_READS_PER_CYCLE);
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
                assert!(write_records.len() < CPU_MAX_WRITES_PER_CYCLE);
                write_records.push(self.memory_chip.borrow_mut().write_cell(
                    $addr_space,
                    $pointer,
                    $data,
                ));
            }};
        }

        let mut public_value_flags = vec![F::zero(); num_public_values];

        let streams_and_metrics = self.streams_and_metrics.as_mut().unwrap();
        let hint_stream = &mut streams_and_metrics.hint_stream;

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
                let hint = match streams_and_metrics.input_stream.pop_front() {
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
            CT_START => {
                let collected_metrics = streams_and_metrics.collected_metrics.clone();
                streams_and_metrics
                    .cycle_tracker
                    .start(debug, collected_metrics);
            }
            CT_END => {
                let collected_metrics = streams_and_metrics.collected_metrics.clone();
                streams_and_metrics
                    .cycle_tracker
                    .end(debug, collected_metrics);
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
                    .map_or_else(CpuMemoryAccessCols::disabled, |read| {
                        CpuMemoryAccessCols::from_read_record(read.clone())
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
                    .map_or_else(CpuMemoryAccessCols::disabled, |write| {
                        CpuMemoryAccessCols::from_write_record(write.clone())
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
            for other_opcode in CORE_INSTRUCTIONS {
                operation_flags.insert(other_opcode, F::from_bool(other_opcode == opcode));
            }

            let is_equal_vec_cols = LocalTraceInstructions::generate_trace_row(
                &IsEqualVecAir::new(WORD_SIZE),
                (vec![read_cols[0].value], vec![read_cols[1].value]),
            );

            let read0_equals_read1 = is_equal_vec_cols.io.is_equal;
            let is_equal_vec_aux = is_equal_vec_cols.aux;

            let aux = CpuAuxCols {
                operation_flags,
                public_value_flags,
                reads: read_cols,
                writes: write_cols,
                read0_equals_read1,
                is_equal_vec_aux,
                reads_aux_cols,
                writes_aux_cols,
            };

            let cols = CpuCols { io, aux };
            self.rows.push(cols.flatten());
        }

        // Update CPU chip state with all changes from this segment.
        self.set_state(CpuState {
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
}

impl<F: PrimeField32> CpuChip<F> {
    /// Pad with NOP rows.
    pub fn pad_rows(&mut self) {
        let curr_height = self.rows.len();
        let correct_height = self.rows.len().next_power_of_two();
        for _ in 0..correct_height - curr_height {
            self.rows.push(self.make_blank_row().flatten());
        }
    }

    /// This must be called for each blank row and results should never be cloned; see [CpuCols::nop_row].
    fn make_blank_row(&self) -> CpuCols<F> {
        let pc = F::from_canonical_usize(self.state.pc);
        let timestamp = F::from_canonical_usize(self.state.timestamp);
        CpuCols::nop_row(self, pc, timestamp)
    }
}

impl<F: PrimeField32> MachineChip<F> for CpuChip<F> {
    fn generate_trace(mut self) -> RowMajorMatrix<F> {
        self.pad_rows();

        RowMajorMatrix::new(self.rows.concat(), CpuCols::<F>::get_width(&self.air))
    }

    fn air<SC: StarkGenericConfig>(&self) -> Box<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        Box::new(self.air.clone())
    }

    fn generate_public_values(&mut self) -> Vec<F> {
        let first_row_pc = self.start_state.pc;
        let last_row_pc = self.state.pc;
        let mut result = vec![
            F::from_canonical_usize(first_row_pc),
            F::from_canonical_usize(last_row_pc),
        ];
        result.extend(self.public_values.iter().map(|pv| pv.unwrap_or(F::zero())));
        result
    }

    fn current_trace_height(&self) -> usize {
        self.rows.len()
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}
