use std::{
    array::from_fn,
    collections::{BTreeMap, HashMap},
    error::Error,
    fmt::Display,
};

use p3_field::{Field, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;

use afs_chips::{
    is_equal_vec::IsEqualVecAir, is_zero::IsZeroAir, sub_chip::LocalTraceInstructions,
};

use crate::{field_arithmetic::FieldArithmeticAir, memory::OpType};

use super::{
    columns::{CpuAuxCols, CpuCols, CpuIoCols, MemoryAccessCols},
    compose, decompose, CpuAir, CpuOptions,
    OpCode::{self, *},
    INST_WIDTH, MAX_READS_PER_CYCLE, MAX_WRITES_PER_CYCLE,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq, derive_new::new)]
pub struct Instruction<F> {
    pub opcode: OpCode,
    pub op_a: F,
    pub op_b: F,
    pub op_c: F,
    pub d: F,
    pub e: F,
}

impl<F: PrimeField64> ArithmeticOperation<F> {
    pub fn from_isize(opcode: OpCode, operand1: isize, operand2: isize, result: isize) -> Self {
        Self {
            opcode,
            operand1: isize_to_field::<F>(operand1),
            operand2: isize_to_field::<F>(operand2),
            result: isize_to_field::<F>(result),
        }
    }

    pub fn to_vec(&self) -> Vec<F> {
        vec![
            F::from_canonical_usize(self.opcode as usize),
            self.operand1,
            self.operand2,
            self.result,
        ]
    }
}

pub fn isize_to_field<F: PrimeField64>(value: isize) -> F {
    if value < 0 {
        return F::neg_one() * F::from_canonical_usize(value.unsigned_abs());
    }
    F::from_canonical_usize(value as usize)
}

impl<F: PrimeField64> Instruction<F> {
    pub fn from_isize(
        opcode: OpCode,
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
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MemoryAccess<const WORD_SIZE: usize, F> {
    pub timestamp: usize,
    pub op_type: OpType,
    pub address_space: F,
    pub address: F,
    pub data: [F; WORD_SIZE],
}

fn memory_access_to_cols<const WORD_SIZE: usize, F: PrimeField64>(
    access: Option<&MemoryAccess<WORD_SIZE, F>>,
) -> MemoryAccessCols<WORD_SIZE, F> {
    let (enabled, address_space, address, value) = match access {
        Some(&MemoryAccess {
            address_space,
            address,
            data,
            ..
        }) => (F::one(), address_space, address, data),
        None => (F::zero(), F::one(), F::zero(), [F::zero(); WORD_SIZE]),
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
        data: value,
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ArithmeticOperation<F> {
    pub opcode: OpCode,
    pub operand1: F,
    pub operand2: F,
    pub result: F,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FieldExtensionOperation<F> {
    pub opcode: OpCode,
    pub operand1: [F; 4],
    pub operand2: [F; 4],
    pub result: [F; 4],
}

impl<F: Field> FieldExtensionOperation<F> {
    pub fn to_vec(&self) -> Vec<F> {
        let mut vec = vec![F::from_canonical_usize(self.opcode as usize)];
        vec.extend(self.operand1.iter());
        vec.extend(self.operand2.iter());
        vec.extend(self.result.iter());
        vec
    }
}

pub struct ProgramExecution<const WORD_SIZE: usize, F> {
    pub program: Vec<Instruction<F>>,
    pub trace_rows: Vec<CpuCols<WORD_SIZE, F>>,
    pub execution_frequencies: Vec<F>,
    pub memory_accesses: Vec<MemoryAccess<WORD_SIZE, F>>,
    pub arithmetic_ops: Vec<ArithmeticOperation<F>>,
}

impl<const WORD_SIZE: usize, F: PrimeField64> ProgramExecution<WORD_SIZE, F> {
    pub fn trace(&self, options: CpuOptions) -> RowMajorMatrix<F> {
        let rows: Vec<F> = self
            .trace_rows
            .iter()
            .flat_map(|row| row.flatten(options))
            .collect();
        RowMajorMatrix::new(rows, CpuCols::<WORD_SIZE, F>::get_width(options))
    }
}

struct Memory<const WORD_SIZE: usize, F> {
    data: HashMap<F, HashMap<F, [F; WORD_SIZE]>>,
    log: Vec<MemoryAccess<WORD_SIZE, F>>,
    clock_cycle: usize,
    reads_this_cycle: Vec<MemoryAccess<WORD_SIZE, F>>,
    writes_this_cycle: Vec<MemoryAccess<WORD_SIZE, F>>,
}

impl<const WORD_SIZE: usize, F: PrimeField64> Memory<WORD_SIZE, F> {
    fn new() -> Self {
        let mut data = HashMap::new();
        data.insert(F::one(), HashMap::new());
        data.insert(F::two(), HashMap::new());

        Self {
            data,
            log: vec![],
            clock_cycle: 0,
            reads_this_cycle: vec![],
            writes_this_cycle: vec![],
        }
    }

    fn read(&mut self, address_space: F, address: F) -> [F; WORD_SIZE] {
        let data = if address_space == F::zero() {
            decompose::<WORD_SIZE, F>(address)
        } else {
            *self.data[&address_space]
                .get(&address)
                .unwrap_or(&[F::zero(); WORD_SIZE])
        };
        let read = MemoryAccess {
            timestamp: ((MAX_READS_PER_CYCLE + MAX_WRITES_PER_CYCLE) * self.clock_cycle)
                + self.reads_this_cycle.len(),
            op_type: OpType::Read,
            address_space,
            address,
            data,
        };
        if read.address_space != F::zero() {
            self.log.push(read);
        }
        self.reads_this_cycle.push(read);
        data
    }

    fn write(&mut self, address_space: F, address: F, data: [F; WORD_SIZE]) {
        if address_space == F::zero() {
            panic!("Attempted to write to address space 0");
        } else {
            let write = MemoryAccess {
                timestamp: ((MAX_READS_PER_CYCLE + MAX_WRITES_PER_CYCLE) * self.clock_cycle)
                    + MAX_READS_PER_CYCLE
                    + self.writes_this_cycle.len(),
                op_type: OpType::Write,
                address_space,
                address,
                data,
            };
            self.log.push(write);
            self.writes_this_cycle.push(write);

            self.data
                .get_mut(&address_space)
                .unwrap()
                .insert(address, data);
        }
    }

    fn complete_clock_cycle(
        &mut self,
    ) -> (
        Vec<MemoryAccess<WORD_SIZE, F>>,
        Vec<MemoryAccess<WORD_SIZE, F>>,
    ) {
        self.clock_cycle += 1;
        let reads = std::mem::take(&mut self.reads_this_cycle);
        let writes = std::mem::take(&mut self.writes_this_cycle);
        (reads, writes)
    }
}

#[derive(Debug)]
pub enum ExecutionError {
    Fail(usize),
    PcOutOfBounds(usize, usize),
    DisabledOperation(OpCode),
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
            ExecutionError::DisabledOperation(op) => write!(f, "opcode {:?} was not enabled", op),
        }
    }
}

impl Error for ExecutionError {}

impl<const WORD_SIZE: usize> CpuAir<WORD_SIZE> {
    pub fn generate_program_execution<F: PrimeField64>(
        &self,
        program: Vec<Instruction<F>>,
    ) -> Result<ProgramExecution<WORD_SIZE, F>, ExecutionError> {
        let mut rows = vec![];
        let mut execution_frequencies = vec![F::zero(); program.len()];
        let mut arithmetic_operations = vec![];

        let mut clock_cycle: usize = 0;
        let mut pc = F::zero();

        let mut memory = Memory::new();

        loop {
            let pc_usize = pc.as_canonical_u64() as usize;
            execution_frequencies[pc_usize] += F::one();

            let instruction = program[pc_usize];
            let opcode = instruction.opcode;
            let a = instruction.op_a;
            let b = instruction.op_b;
            let c = instruction.op_c;
            let d = instruction.d;
            let e = instruction.e;

            let io = CpuIoCols {
                clock_cycle: F::from_canonical_usize(clock_cycle),
                pc,
                opcode: F::from_canonical_usize(opcode as usize),
                op_a: a,
                op_b: b,
                op_c: c,
                d,
                e,
            };

            let mut next_pc = pc + F::one();

            match opcode {
                // d[a] <- e[d[c] + b]
                LOADW => {
                    let base_pointer = compose(memory.read(d, c));
                    let value = memory.read(e, base_pointer + b);
                    memory.write(d, a, value);
                }
                // e[d[c] + b] <- d[a]
                STOREW => {
                    let base_pointer = compose(memory.read(d, c));
                    let value = memory.read(d, a);
                    memory.write(e, base_pointer + b, value);
                }
                // d[a] <- pc + INST_WIDTH, pc <- pc + b
                JAL => {
                    memory.write(d, a, decompose(pc + F::from_canonical_usize(INST_WIDTH)));
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
                TERMINATE => {
                    next_pc = pc;
                }
                opcode @ (FADD | FSUB | FMUL | FDIV) => {
                    if self.options.field_arithmetic_enabled {
                        // read from d[b] and e[c]
                        let operand1 = compose(memory.read(d, b));
                        let operand2 = compose(memory.read(e, c));
                        // write to d[a]
                        let result =
                            FieldArithmeticAir::solve(opcode, (operand1, operand2)).unwrap();
                        memory.write(d, a, decompose(result));

                        arithmetic_operations.push(ArithmeticOperation {
                            opcode,
                            operand1,
                            operand2,
                            result,
                        });
                    } else {
                        return Err(ExecutionError::DisabledOperation(opcode));
                    }
                }
                FAIL => return Err(ExecutionError::Fail(pc_usize)),
                PRINTF => {
                    let value = memory.read(d, a);
                    println!("{}", compose(value));
                }
            };

            let mut operation_flags = BTreeMap::new();
            for other_opcode in self.options.enabled_instructions() {
                operation_flags.insert(other_opcode, F::from_bool(other_opcode == opcode));
            }

            // complete the clock cycle and get the read and write cols
            let (reads, writes) = memory.complete_clock_cycle();
            assert!(reads.len() <= MAX_READS_PER_CYCLE);
            assert!(writes.len() <= MAX_WRITES_PER_CYCLE);

            let accesses = from_fn(|i| {
                memory_access_to_cols(if i < MAX_READS_PER_CYCLE {
                    reads.get(i)
                } else {
                    writes.get(i - MAX_READS_PER_CYCLE)
                })
            });

            let is_equal_vec_cols = LocalTraceInstructions::generate_trace_row(
                &IsEqualVecAir::new(WORD_SIZE),
                (accesses[0].data.to_vec(), accesses[1].data.to_vec()),
            );

            let read0_equals_read1 = is_equal_vec_cols.io.prod;
            let is_equal_vec_aux = is_equal_vec_cols.aux;

            let aux = CpuAuxCols {
                operation_flags,
                accesses,
                read0_equals_read1,
                is_equal_vec_aux,
            };

            let cols = CpuCols { io, aux };
            rows.push(cols);

            pc = next_pc;
            clock_cycle += 1;

            if opcode == TERMINATE && rows.len().is_power_of_two() {
                break;
            }
        }

        Ok(ProgramExecution {
            program,
            execution_frequencies,
            trace_rows: rows,
            memory_accesses: memory.log,
            arithmetic_ops: arithmetic_operations,
        })
    }
}
