use std::collections::{HashMap, VecDeque};

use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use afs_chips::{is_equal::IsEqualAir, is_zero::IsZeroAir, sub_chip::LocalTraceInstructions};

use super::{
    columns::{CPUAuxCols, CPUCols, CPUIOCols, MemoryAccessCols},
    CPUChip, OpCode,
    OpCode::*,
    INST_WIDTH,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Instruction<F> {
    pub opcode: OpCode,
    pub op_a: F,
    pub op_b: F,
    pub op_c: F,
    pub as_b: F,
    pub as_c: F,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MemoryAccess<F> {
    pub clock: usize,
    pub is_write: bool,
    pub address_space: F,
    pub address: F,
    pub value: F,
}

fn memory_access_to_cols<F: PrimeField64>(access: Option<MemoryAccess<F>>) -> MemoryAccessCols<F> {
    let (enabled, address_space, address, value) = match access {
        Some(MemoryAccess {
            clock: _,
            is_write: _,
            address_space,
            address,
            value,
        }) => (F::one(), address_space, address, value),
        None => (F::zero(), F::one(), F::zero(), F::zero()),
    };
    let is_zero_cols = LocalTraceInstructions::generate_trace_row(&IsZeroAir {}, address_space);
    let is_immediate = is_zero_cols.io.is_zero;
    let is_zero_aux = is_zero_cols.inv;
    MemoryAccessCols {
        enabled,
        address_space,
        is_immediate,
        is_zero_aux,
        address,
        value,
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ArithmeticOperation<F> {
    pub opcode: OpCode,
    pub operand1: F,
    pub operand2: F,
    pub result: F,
}

pub struct ProgramExecution<F> {
    pub program: Vec<Instruction<F>>,
    pub trace_rows: Vec<CPUCols<F>>,
    pub execution_frequencies: Vec<F>,
    pub memory_accesses: Vec<MemoryAccess<F>>,
    pub arithmetic_ops: Vec<ArithmeticOperation<F>>,
}

impl<F: PrimeField64> ProgramExecution<F> {
    pub fn trace(&self) -> RowMajorMatrix<F> {
        let rows: Vec<F> = self
            .trace_rows
            .iter()
            .flat_map(|row| row.flatten())
            .collect();
        let num_cols = rows.len() / self.trace_rows.len();
        RowMajorMatrix::new(rows, num_cols)
    }
}

struct Memory<F> {
    data: HashMap<F, HashMap<F, F>>,
    log: Vec<MemoryAccess<F>>,
    clock_cycle: usize,
    reads_this_cycle: VecDeque<MemoryAccess<F>>,
    writes_this_cycle: VecDeque<MemoryAccess<F>>,
}

impl<F: PrimeField64> Memory<F> {
    fn new() -> Self {
        let mut data = HashMap::new();
        data.insert(F::one(), HashMap::new());
        data.insert(F::two(), HashMap::new());

        Self {
            data,
            log: vec![],
            clock_cycle: 0,
            reads_this_cycle: VecDeque::new(),
            writes_this_cycle: VecDeque::new(),
        }
    }

    fn read(&mut self, address_space: F, address: F) -> F {
        let value = if address_space == F::zero() {
            address
        } else {
            *self.data[&address_space]
                .get(&address)
                .unwrap_or(&F::zero())
        };
        let read = MemoryAccess {
            clock: self.clock_cycle,
            is_write: false,
            address_space,
            address,
            value,
        };
        if read.address_space != F::zero() {
            self.log.push(read);
        }
        self.reads_this_cycle.push_back(read);
        value
    }

    fn write(&mut self, address_space: F, address: F, value: F) {
        if address_space == F::zero() {
            panic!("Attempted to write to address space 0");
        } else {
            let write = MemoryAccess {
                clock: self.clock_cycle,
                is_write: true,
                address_space,
                address,
                value,
            };
            self.log.push(write);
            self.writes_this_cycle.push_back(write);

            self.data
                .get_mut(&address_space)
                .unwrap()
                .insert(address, value);
        }
    }

    fn complete_clock_cycle(&mut self) -> (VecDeque<MemoryAccess<F>>, VecDeque<MemoryAccess<F>>) {
        self.clock_cycle += 1;
        let reads = std::mem::take(&mut self.reads_this_cycle);
        let writes = std::mem::take(&mut self.writes_this_cycle);
        (reads, writes)
    }
}

impl CPUChip {
    pub fn generate_trace<F: PrimeField64>(
        &self,
        program: Vec<Instruction<F>>,
    ) -> ProgramExecution<F> {
        let mut rows = vec![];
        let mut execution_frequencies = vec![F::zero(); program.len()];
        let mut arithmetic_operations = vec![];

        let mut clock_cycle: usize = 0;
        let mut pc = F::zero();

        let mut memory = Memory::new();

        loop {
            let pc_usize = pc.as_canonical_u64() as usize;
            if pc_usize >= program.len() {
                break;
            }
            execution_frequencies[pc_usize] += F::one();

            let instruction = program[pc.as_canonical_u64() as usize];
            let opcode = instruction.opcode;
            let a = instruction.op_a;
            let b = instruction.op_b;
            let c = instruction.op_c;
            let d = instruction.as_b;
            let e = instruction.as_c;

            let io = CPUIOCols {
                clock_cycle: F::from_canonical_usize(clock_cycle),
                pc,
                opcode: F::from_canonical_usize(opcode as usize),
                op_a: a,
                op_b: b,
                op_c: c,
                as_b: d,
                as_c: e,
            };

            let mut operation_flags = vec![F::zero(); self.air.options.num_operations()];
            operation_flags[opcode as usize] = F::one();

            let mut next_pc = pc + F::one();

            match opcode {
                // d[a] <- e[d[c] + b]
                LOADW => {
                    let base_pointer = memory.read(d, c);
                    let value = memory.read(e, base_pointer + b);
                    memory.write(d, a, value);
                }
                // e[d[c] + b] <- d[a]
                STOREW => {
                    let base_pointer = memory.read(d, c);
                    let value = memory.read(d, a);
                    memory.write(e, base_pointer + b, value);
                }
                // d[a] <- pc + INST_WIDTH, pc <- pc + b
                JAL => {
                    memory.write(d, a, pc + F::from_canonical_usize(INST_WIDTH));
                    next_pc = pc + b;
                }
                // If d[a] = e[b], pc <- pc + c
                BEQ => {
                    let left = memory.read(d, a);
                    let right = memory.read(e, b);
                    if left == right {
                        next_pc = pc + c;
                    }
                }
                // If d[a] != e[b], pc <- pc + c
                BNE => {
                    let left = memory.read(d, a);
                    let right = memory.read(e, b);
                    if left != right {
                        next_pc = pc + c;
                    }
                }
                opcode @ (FADD | FSUB | FMUL | FDIV) => {
                    if self.air.options.field_arithmetic_enabled {
                        // read from e[b] and e[c]
                        let operand1 = memory.read(e, b);
                        let operand2 = memory.read(e, c);
                        let result = match opcode {
                            FADD => operand1 + operand2,
                            FSUB => operand1 - operand2,
                            FMUL => operand1 * operand2,
                            FDIV => operand1 / operand2,
                            _ => unreachable!(),
                        };
                        // write to d[a]
                        memory.write(d, a, result);

                        arithmetic_operations.push(ArithmeticOperation {
                            opcode,
                            operand1,
                            operand2,
                            result,
                        });
                    } else {
                        panic!("Field arithmetic is not enabled");
                    }
                }
            };

            // complete the clock cycle and get the read and write cols
            let (mut read_cols, mut write_cols) = memory.complete_clock_cycle();
            let read1 = memory_access_to_cols(read_cols.pop_front());
            let read2 = memory_access_to_cols(read_cols.pop_front());
            let write = memory_access_to_cols(write_cols.pop_front());

            if !read_cols.is_empty() {
                panic!("Too many reads");
            }
            if !write_cols.is_empty() {
                panic!("Too many writes");
            }

            let is_equal_cols = LocalTraceInstructions::generate_trace_row(
                &IsEqualAir {},
                (read1.value, read2.value),
            );
            let beq_check = is_equal_cols.io.is_equal;
            let is_equal_aux = is_equal_cols.aux.inv;

            let aux = CPUAuxCols {
                operation_flags,
                read1,
                read2,
                write,
                beq_check,
                is_equal_aux,
            };

            let cols = CPUCols { io, aux };
            rows.push(cols);

            pc = next_pc;
            clock_cycle += 1;
        }

        ProgramExecution {
            program,
            execution_frequencies,
            trace_rows: rows,
            memory_accesses: memory.log,
            arithmetic_ops: arithmetic_operations,
        }
    }
}
