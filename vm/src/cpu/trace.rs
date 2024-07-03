use std::{collections::BTreeMap, error::Error, fmt::Display};

use p3_field::{Field, PrimeField32, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;

use afs_chips::{
    is_equal_vec::IsEqualVecAir, is_zero::IsZeroAir, sub_chip::LocalTraceInstructions,
};

use crate::memory::{compose, decompose};
use crate::vm::VirtualMachine;

use super::{
    columns::{CpuAuxCols, CpuCols, CpuIoCols, MemoryAccessCols},
    CpuAir,
    OpCode::{self, *},
    INST_WIDTH, MAX_ACCESSES_PER_CYCLE, MAX_READS_PER_CYCLE, MAX_WRITES_PER_CYCLE,
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
pub fn isize_to_field<F: Field>(value: isize) -> F {
    if value < 0 {
        return F::neg_one() * F::from_canonical_usize(value.unsigned_abs());
    }
    F::from_canonical_usize(value as usize)
}

impl<F: Field> Instruction<F> {
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

fn disabled_memory_cols<const WORD_SIZE: usize, F: PrimeField64>() -> MemoryAccessCols<WORD_SIZE, F>
{
    memory_access_to_cols(false, F::one(), F::zero(), [F::zero(); WORD_SIZE])
}

fn memory_access_to_cols<const WORD_SIZE: usize, F: PrimeField64>(
    enabled: bool,
    address_space: F,
    address: F,
    data: [F; WORD_SIZE],
) -> MemoryAccessCols<WORD_SIZE, F> {
    let is_zero_cols = LocalTraceInstructions::generate_trace_row(&IsZeroAir {}, address_space);
    let is_immediate = is_zero_cols.io.is_zero;
    let is_zero_aux = is_zero_cols.inv;
    MemoryAccessCols {
        enabled: F::from_bool(enabled),
        address_space,
        is_immediate,
        is_zero_aux,
        address,
        data,
    }
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
    pub fn generate_trace<F: PrimeField32>(
        vm: &mut VirtualMachine<WORD_SIZE, F>,
    ) -> Result<RowMajorMatrix<F>, ExecutionError> {
        let mut rows = vec![];

        let mut clock_cycle: usize = 0;
        let mut pc = F::zero();

        loop {
            let pc_usize = pc.as_canonical_u64() as usize;

            let instruction = vm.program_chip.get_instruction(pc_usize);
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

            let mut accesses = [disabled_memory_cols(); MAX_ACCESSES_PER_CYCLE];
            let mut num_reads = 0;
            let mut num_writes = 0;

            macro_rules! read {
                ($address_space: expr, $address: expr) => {{
                    num_reads += 1;
                    assert!(num_reads <= MAX_READS_PER_CYCLE);
                    let timestamp = (MAX_ACCESSES_PER_CYCLE * clock_cycle) + (num_reads - 1);
                    let data = vm
                        .memory_chip
                        .read_word(timestamp, $address_space, $address);
                    accesses[num_reads - 1] =
                        memory_access_to_cols(true, $address_space, $address, data);
                    compose(data)
                }};
            }

            macro_rules! write {
                ($address_space: expr, $address: expr, $data: expr) => {{
                    num_writes += 1;
                    assert!(num_writes <= MAX_WRITES_PER_CYCLE);
                    let timestamp = (MAX_ACCESSES_PER_CYCLE * clock_cycle)
                        + (MAX_READS_PER_CYCLE + num_writes - 1);
                    let word = decompose($data);
                    vm.memory_chip
                        .write_word(timestamp, $address_space, $address, word);
                    accesses[MAX_READS_PER_CYCLE + num_writes - 1] =
                        memory_access_to_cols(true, $address_space, $address, word);
                }};
            }

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
                TERMINATE => {
                    next_pc = pc;
                }
                opcode @ (FADD | FSUB | FMUL | FDIV) => {
                    if vm.options().field_arithmetic_enabled {
                        // read from d[b] and e[c]
                        let operand1 = read!(d, b);
                        let operand2 = read!(e, c);
                        // write to d[a]
                        let result = vm
                            .field_arithmetic_chip
                            .calculate(opcode, (operand1, operand2));
                        write!(d, a, result);
                    } else {
                        return Err(ExecutionError::DisabledOperation(opcode));
                    }
                }
                FAIL => return Err(ExecutionError::Fail(pc_usize)),
                PRINTF => {
                    let value = read!(d, a);
                    println!("{}", value);
                }
            };

            let mut operation_flags = BTreeMap::new();
            for other_opcode in vm.options().enabled_instructions() {
                operation_flags.insert(other_opcode, F::from_bool(other_opcode == opcode));
            }

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
            rows.extend(cols.flatten(vm.options()));

            pc = next_pc;
            clock_cycle += 1;

            if opcode == TERMINATE && clock_cycle.is_power_of_two() {
                break;
            }
        }

        Ok(RowMajorMatrix::new(
            rows,
            CpuCols::<WORD_SIZE, F>::get_width(vm.options()),
        ))
    }
}
