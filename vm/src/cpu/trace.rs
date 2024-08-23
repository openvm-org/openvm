use std::{
    collections::{BTreeMap, VecDeque},
    error::Error,
    fmt::Display,
};

use itertools::Itertools;
use p3_air::BaseAir;
use p3_commit::PolynomialSpace;
use p3_field::{Field, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig};

use afs_primitives::{is_equal_vec::IsEqualVecAir, sub_chip::LocalTraceInstructions};
use afs_stark_backend::rap::AnyRap;

use crate::{
    arch::{
        chips::{InstructionExecutor, MachineChip},
        columns::{ExecutionState, NUM_OPERANDS},
        instructions::{CORE_INSTRUCTIONS, Opcode, Opcode::*},
    },
    cpu::trace::ExecutionError::{PublicValueIndexOutOfBounds, PublicValueNotEqual},
    memory::{
        compose, decompose,
        manager::{MemoryManager, operation::MemoryOperation},
        OpType,
    },
    vm::ExecutionSegment,
};

use super::{
    columns::{CpuAuxCols, CpuCols, CpuIoCols},
    CPU_MAX_ACCESSES_PER_CYCLE, CPU_MAX_READS_PER_CYCLE, CPU_MAX_WRITES_PER_CYCLE, CpuChip, CpuState,
    INST_WIDTH, timestamp_delta,
};

#[allow(clippy::too_many_arguments)]
#[derive(Clone, Debug, PartialEq, Eq, derive_new::new)]
pub struct Instruction<F> {
    pub opcode: Opcode,
    pub op_a: F,
    pub op_b: F,
    pub op_c: F,
    pub d: F,
    pub e: F,
    pub op_f: F,
    pub op_g: F,
    pub debug: String,
}

pub fn isize_to_field<F: Field>(value: isize) -> F {
    if value < 0 {
        return F::neg_one() * F::from_canonical_usize(value.unsigned_abs());
    }
    F::from_canonical_usize(value as usize)
}

impl<F: Field> Instruction<F> {
    #[allow(clippy::too_many_arguments)]
    pub fn from_isize(
        opcode: Opcode,
        op_a: isize,
        op_b: isize,
        op_c: isize,
        d: isize,
        e: isize,
    ) -> Self {
        Self {
            opcode,
            op_a: isize_to_field::<F>(op_a),
            op_b: isize_to_field::<F>(op_b),
            op_c: isize_to_field::<F>(op_c),
            d: isize_to_field::<F>(d),
            e: isize_to_field::<F>(e),
            op_f: isize_to_field::<F>(0),
            op_g: isize_to_field::<F>(0),
            debug: String::new(),
        }
    }

    pub fn from_usize<const N: usize>(opcode: Opcode, operands: [usize; N]) -> Self {
        let mut operands = operands.to_vec();
        while operands.len() < NUM_OPERANDS {
            operands.push(0);
        }
        let operands = operands
            .into_iter()
            .map(F::from_canonical_usize)
            .collect_vec();
        Self {
            opcode,
            op_a: operands[0],
            op_b: operands[1],
            op_c: operands[2],
            d: operands[3],
            e: operands[4],
            op_f: operands[5],
            op_g: operands[6],
            debug: String::new(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn large_from_isize(
        opcode: Opcode,
        op_a: isize,
        op_b: isize,
        op_c: isize,
        d: isize,
        e: isize,
        op_f: isize,
        op_g: isize,
    ) -> Self {
        Self {
            opcode,
            op_a: isize_to_field::<F>(op_a),
            op_b: isize_to_field::<F>(op_b),
            op_c: isize_to_field::<F>(op_c),
            d: isize_to_field::<F>(d),
            e: isize_to_field::<F>(e),
            op_f: isize_to_field::<F>(op_f),
            op_g: isize_to_field::<F>(op_g),
            debug: String::new(),
        }
    }

    pub fn debug(opcode: Opcode, debug: &str) -> Self {
        Self {
            opcode,
            op_a: F::zero(),
            op_b: F::zero(),
            op_c: F::zero(),
            d: F::zero(),
            e: F::zero(),
            op_f: F::zero(),
            op_g: F::zero(),
            debug: String::from(debug),
        }
    }
}

impl<T: Default> Default for Instruction<T> {
    fn default() -> Self {
        Self {
            opcode: NOP,
            op_a: T::default(),
            op_b: T::default(),
            op_c: T::default(),
            d: T::default(),
            e: T::default(),
            op_f: T::default(),
            op_g: T::default(),
            debug: String::new(),
        }
    }
}

#[derive(Debug)]
pub enum ExecutionError {
    Fail(usize),
    PcOutOfBounds(usize, usize),
    DisabledOperation(usize, Opcode),
    HintOutOfBounds(usize),
    EndOfInputStream(usize),
    PublicValueIndexOutOfBounds(usize, usize, usize),
    PublicValueNotEqual(usize, usize, usize, usize),
}

impl Display for ExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionError::Fail(pc) => write!(f, "execution failed at pc = {}", pc),
            ExecutionError::PcOutOfBounds(pc, program_len) => write!(
                f,
                "pc = {} out of bounds for program of length {}",
                pc, program_len
            ),
            ExecutionError::DisabledOperation(pc, op) => {
                write!(f, "at pc = {}, opcode {:?} was not enabled", pc, op)
            }
            ExecutionError::HintOutOfBounds(pc) => write!(f, "at pc = {}", pc),
            ExecutionError::EndOfInputStream(pc) => write!(f, "at pc = {}", pc),
            ExecutionError::PublicValueIndexOutOfBounds(
                pc,
                num_public_values,
                public_value_index,
            ) => write!(
                f,
                "at pc = {}, tried to publish into index {} when num_public_values = {}",
                pc, public_value_index, num_public_values
            ),
            ExecutionError::PublicValueNotEqual(
                pc,
                public_value_index,
                existing_value,
                new_value,
            ) => write!(
                f,
                "at pc = {}, tried to publish value {} into index {}, but already had {}",
                pc, new_value, public_value_index, existing_value
            ),
        }
    }
}

impl Error for ExecutionError {}

impl<const WORD_SIZE: usize, F: PrimeField32> CpuChip<WORD_SIZE, F> {
    pub fn execute(vm: &mut ExecutionSegment<F>) -> Result<(), ExecutionError> {
        let mut clock_cycle: usize = vm.cpu_chip.borrow().state.clock_cycle;
        let mut timestamp: usize = vm.cpu_chip.borrow().state.timestamp;
        let mut pc = F::from_canonical_usize(vm.cpu_chip.borrow().state.pc);

        let mut hint_stream = vm.hint_stream.clone();
        let mut cycle_tracker = std::mem::take(&mut vm.cycle_tracker);
        let mut is_done = false;
        let mut collect_metrics = vm.config.collect_metrics;

        let cpu_options = vm.cpu_chip.borrow().air.options;
        let num_public_values = cpu_options.num_public_values;

        loop {
            let pc_usize = pc.as_canonical_u64() as usize;

            let (instruction, debug_info) =
                vm.program_chip.borrow_mut().get_instruction(pc_usize)?;

            let dsl_instr = match debug_info {
                Some(debug_info) => debug_info.dsl_instruction,
                None => String::new(),
            };

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

            let mut mem_ops: [_; CPU_MAX_ACCESSES_PER_CYCLE] =
                core::array::from_fn(|_| MemoryOperation::<1, F>::default());
            let mut mem_read_trace_builder =
                MemoryManager::make_trace_builder(vm.memory_manager.clone());
            let mut mem_write_trace_builder =
                MemoryManager::make_trace_builder(vm.memory_manager.clone());
            let mut num_reads = 0;
            let mut num_writes = 0;

            let prev_trace_cells = vm.current_trace_cells();

            macro_rules! read {
                ($addr_space: expr, $pointer: expr) => {{
                    num_reads += 1;
                    assert!(num_reads <= CPU_MAX_READS_PER_CYCLE);

                    mem_ops[num_reads - 1] =
                        mem_read_trace_builder.read_word($addr_space, $pointer);
                    compose(mem_ops[num_reads - 1].cell.data)
                }};
            }

            macro_rules! disabled_read {
                () => {{
                    num_reads += 1;
                    assert!(num_reads <= CPU_MAX_READS_PER_CYCLE);

                    mem_ops[num_reads - 1] =
                        mem_read_trace_builder.disabled_op(F::zero(), OpType::Read);
                }};
            }

            macro_rules! write {
                ($addr_space: expr, $pointer: expr, $data: expr) => {{
                    // First, finalize the read accesses
                    while num_reads < CPU_MAX_READS_PER_CYCLE {
                        disabled_read!();
                    }

                    num_writes += 1;
                    assert!(num_writes <= CPU_MAX_WRITES_PER_CYCLE);

                    let word = decompose($data);
                    mem_ops[CPU_MAX_READS_PER_CYCLE + num_writes - 1] =
                        mem_write_trace_builder.write_word($addr_space, $pointer, word);
                }};
            }

            macro_rules! generate_disabled_ops {
                () => {{
                    while num_reads < CPU_MAX_READS_PER_CYCLE {
                        disabled_read!();
                    }

                    while num_writes < CPU_MAX_WRITES_PER_CYCLE {
                        num_writes += 1;
                        mem_ops[CPU_MAX_READS_PER_CYCLE + num_writes - 1] =
                            mem_write_trace_builder.disabled_op(F::zero(), OpType::Write);
                    }
                }};
            }

            if opcode == FAIL {
                return Err(ExecutionError::Fail(pc_usize));
            }

            let mut public_value_flags = vec![F::zero(); num_public_values];

            if vm.executors.contains_key(&opcode) {
                generate_disabled_ops!();

                let executor = vm.executors.get_mut(&opcode).unwrap();
                let next_state = InstructionExecutor::execute(
                    executor,
                    &instruction,
                    ExecutionState::new(pc_usize, timestamp),
                );
                next_pc = F::from_canonical_usize(next_state.pc);
                timestamp = next_state.timestamp;
            } else {
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
                            return Err(PublicValueIndexOutOfBounds(
                                pc_usize,
                                num_public_values,
                                public_value_index,
                            ));
                        }
                        public_value_flags[public_value_index] = F::one();

                        let public_values = &mut vm.cpu_chip.borrow_mut().public_values;
                        match public_values[public_value_index] {
                            None => public_values[public_value_index] = Some(value),
                            Some(exising_value) => {
                                if value != exising_value {
                                    return Err(PublicValueNotEqual(
                                        pc_usize,
                                        public_value_index,
                                        exising_value.as_canonical_u64() as usize,
                                        value.as_canonical_u64() as usize,
                                    ));
                                }
                            }
                        }
                    }
                    /*F_LESS_THAN => {
                        let operand1 = read!(d, b);
                        let operand2 = read!(e, c);
                        let result = vm.is_less_than_chip.compare((operand1, operand2));
                        write!(d, a, result);
                    }*/
                    PRINTF => {
                        let value = read!(d, a);
                        println!("{}", value);
                    }
                    HINT_INPUT => {
                        let hint = match vm.input_stream.pop_front() {
                            Some(hint) => hint,
                            None => return Err(ExecutionError::EndOfInputStream(pc_usize)),
                        };
                        hint_stream = VecDeque::new();
                        hint_stream.push_back(F::from_canonical_usize(hint.len()));
                        hint_stream.extend(hint);
                    }
                    HINT_BITS => {
                        let val = vm.memory_manager.borrow_mut().unsafe_read_word(d, a);
                        let mut val = val[0].as_canonical_u32();

                        hint_stream = VecDeque::new();
                        for _ in 0..32 {
                            hint_stream.push_back(F::from_canonical_u32(val & 1));
                            val >>= 1;
                        }
                    }
                    // e[d[a] + b] <- hint_stream.next()
                    SHINTW => {
                        let hint = match hint_stream.pop_front() {
                            Some(hint) => hint,
                            None => return Err(ExecutionError::HintOutOfBounds(pc_usize)),
                        };
                        let base_pointer = read!(d, a);
                        write!(e, base_pointer + b, hint);
                    }
                    CT_START => cycle_tracker.start(debug, vm.collected_metrics.clone()),
                    CT_END => cycle_tracker.end(debug, vm.collected_metrics.clone()),
                    _ => return Err(ExecutionError::DisabledOperation(pc_usize, opcode)),
                };
                timestamp += timestamp_delta(opcode);
            }

            let now_trace_cells = vm.current_trace_cells();
            let added_trace_cells = now_trace_cells - prev_trace_cells;

            if collect_metrics {
                //vm.update_chip_metrics();
                vm.collected_metrics
                    .opcode_counts
                    .entry(opcode.to_string())
                    .and_modify(|count| *count += 1)
                    .or_insert(1);

                if !dsl_instr.is_empty() {
                    vm.collected_metrics
                        .dsl_counts
                        .entry(dsl_instr)
                        .and_modify(|count| *count += 1)
                        .or_insert(1);
                }

                vm.collected_metrics
                    .opcode_trace_cells
                    .entry(opcode.to_string())
                    .and_modify(|count| *count += added_trace_cells)
                    .or_insert(added_trace_cells);
            }

            // Finalizing memory accesses
            for mem_op in &mut mem_ops[num_reads..CPU_MAX_READS_PER_CYCLE] {
                *mem_op = mem_read_trace_builder.disabled_op(F::zero(), OpType::Read);
            }
            for mem_op in
                &mut mem_ops[CPU_MAX_READS_PER_CYCLE + num_writes..CPU_MAX_ACCESSES_PER_CYCLE]
            {
                *mem_op = mem_write_trace_builder.disabled_op(F::zero(), OpType::Write);
            }

            let mem_oc_aux_cols: Vec<_> = mem_read_trace_builder
                .take_accesses_buffer()
                .into_iter()
                .chain(mem_write_trace_builder.take_accesses_buffer())
                .collect();
            let mem_oc_aux_cols = mem_oc_aux_cols.try_into().unwrap();

            let mut operation_flags = BTreeMap::new();
            for other_opcode in CORE_INSTRUCTIONS {
                operation_flags.insert(other_opcode, F::from_bool(other_opcode == opcode));
            }

            let is_equal_vec_cols = LocalTraceInstructions::generate_trace_row(
                &IsEqualVecAir::new(WORD_SIZE),
                (mem_ops[0].cell.data.to_vec(), mem_ops[1].cell.data.to_vec()),
            );

            let read0_equals_read1 = is_equal_vec_cols.io.is_equal;
            let is_equal_vec_aux = is_equal_vec_cols.aux;

            let aux = CpuAuxCols {
                operation_flags,
                public_value_flags,
                mem_ops,
                read0_equals_read1,
                is_equal_vec_aux,
                mem_oc_aux_cols,
            };

            let cols = CpuCols { io, aux };
            vm.cpu_chip.borrow_mut().rows.push(cols.flatten());

            pc = next_pc;

            clock_cycle += 1;
            if opcode == TERMINATE {
                // Due to row padding, the padded rows will all have opcode TERMINATE, so stop metric collection after the first one
                collect_metrics = false;
            }
            if opcode == TERMINATE
                && vm
                    .cpu_chip
                    .borrow()
                    .current_trace_height()
                    .is_power_of_two()
            {
                is_done = true;
                break;
            }
            if vm.should_segment() {
                break;
            }
        }

        // Update CPU chip state with all changes from this segment.
        vm.cpu_chip.borrow_mut().set_state(CpuState {
            clock_cycle,
            timestamp,
            pc: pc.as_canonical_u64() as usize,
            is_done,
        });
        vm.hint_stream = hint_stream;
        vm.cycle_tracker = cycle_tracker;

        Ok(())
    }

    /// Pad with NOP rows.
    pub fn pad_rows(&mut self) {
        let pc = F::from_canonical_usize(self.state.pc);
        let timestamp = F::from_canonical_usize(self.state.timestamp);
        let nop_row = CpuCols::nop_row(self, pc, timestamp).flatten();
        let correct_len = (self.rows.len() + 1).next_power_of_two();
        self.rows.resize(correct_len, nop_row);
    }
}

impl<const WORD_SIZE: usize, F: PrimeField32> MachineChip<F> for CpuChip<WORD_SIZE, F> {
    fn generate_trace(&mut self) -> RowMajorMatrix<F> {
        if !self.state.is_done {
            self.pad_rows();
        }

        RowMajorMatrix::new(
            self.rows.concat(),
            CpuCols::<WORD_SIZE, F>::get_width(&self.air),
        )
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
