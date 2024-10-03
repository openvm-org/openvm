use std::sync::Arc;

use afs_primitives::xor::lookup::XorLookupChip;
use air::ArithmeticLogicAir;
use p3_field::PrimeField32;

use crate::{
    arch::{
        instructions::{U256Opcode, UsizeOpcode},
        ExecutionBridge, ExecutionBus, ExecutionState, InstructionExecutor,
    },
    memory::{MemoryChipRef, MemoryReadRecord, MemoryWriteRecord},
    program::{bridge::ProgramBus, ExecutionError, Instruction},
};

mod air;
mod bridge;
mod columns;
mod trace;

// pub use air::*;
pub use columns::*;

#[cfg(test)]
mod tests;

pub const ALU_CMP_INSTRUCTIONS: [U256Opcode; 3] = [U256Opcode::LT, U256Opcode::EQ, U256Opcode::SLT];
pub const ALU_ARITHMETIC_INSTRUCTIONS: [U256Opcode; 2] = [U256Opcode::ADD, U256Opcode::SUB];
pub const ALU_BITWISE_INSTRUCTIONS: [U256Opcode; 3] =
    [U256Opcode::XOR, U256Opcode::AND, U256Opcode::OR];

#[derive(Clone, Debug)]
pub enum WriteRecord<T, const NUM_LIMBS: usize> {
    Long(MemoryWriteRecord<T, NUM_LIMBS>),
    Bool(MemoryWriteRecord<T, 1>),
}

#[derive(Clone, Debug)]
pub struct ArithmeticLogicRecord<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub from_state: ExecutionState<usize>,
    pub instruction: Instruction<T>,

    pub x_ptr_read: MemoryReadRecord<T, 1>,
    pub y_ptr_read: MemoryReadRecord<T, 1>,
    pub z_ptr_read: MemoryReadRecord<T, 1>,

    pub x_read: MemoryReadRecord<T, NUM_LIMBS>,
    pub y_read: MemoryReadRecord<T, NUM_LIMBS>,
    pub z_write: WriteRecord<T, NUM_LIMBS>,

    // sign of x and y if SLT, else should be 0
    pub x_sign: T,
    pub y_sign: T,

    // empty if not bool instruction, else contents of this vector will be stored in z
    pub cmp_buffer: Vec<T>,
}

#[derive(Clone, Debug)]
pub struct ArithmeticLogicChip<T: PrimeField32, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub air: ArithmeticLogicAir<NUM_LIMBS, LIMB_BITS>,
    data: Vec<ArithmeticLogicRecord<T, NUM_LIMBS, LIMB_BITS>>,
    memory_chip: MemoryChipRef<T>,
    pub xor_lookup_chip: Arc<XorLookupChip<LIMB_BITS>>,

    offset: usize,
}

impl<T: PrimeField32, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    ArithmeticLogicChip<T, NUM_LIMBS, LIMB_BITS>
{
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_chip: MemoryChipRef<T>,
        xor_lookup_chip: Arc<XorLookupChip<LIMB_BITS>>,
        offset: usize,
    ) -> Self {
        let memory_bridge = memory_chip.borrow().memory_bridge();
        Self {
            air: ArithmeticLogicAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
                bus: xor_lookup_chip.bus(),
                offset,
            },
            data: vec![],
            memory_chip,
            xor_lookup_chip,
            offset,
        }
    }
}

impl<T: PrimeField32, const NUM_LIMBS: usize, const LIMB_BITS: usize> InstructionExecutor<T>
    for ArithmeticLogicChip<T, NUM_LIMBS, LIMB_BITS>
{
    fn execute(
        &mut self,
        instruction: Instruction<T>,
        from_state: ExecutionState<usize>,
    ) -> Result<ExecutionState<usize>, ExecutionError> {
        let Instruction {
            opcode,
            op_a: a,
            op_b: b,
            op_c: c,
            d,
            e,
            ..
        } = instruction.clone();
        let opcode = U256Opcode::from_usize(opcode - self.offset);

        let mut memory_chip = self.memory_chip.borrow_mut();
        debug_assert_eq!(
            from_state.timestamp,
            memory_chip.timestamp().as_canonical_u32() as usize
        );

        let [z_ptr_read, x_ptr_read, y_ptr_read] =
            [a, b, c].map(|ptr_of_ptr| memory_chip.read_cell(d, ptr_of_ptr));
        let x_read = memory_chip.read::<NUM_LIMBS>(e, x_ptr_read.value());
        let y_read = memory_chip.read::<NUM_LIMBS>(e, y_ptr_read.value());

        let x = x_read.data.map(|x| x.as_canonical_u32());
        let y = y_read.data.map(|x| x.as_canonical_u32());
        let (z, cmp) = solve_alu::<T, NUM_LIMBS, LIMB_BITS>(opcode, &x, &y);

        let z_write = if ALU_CMP_INSTRUCTIONS.contains(&opcode) {
            WriteRecord::Bool(memory_chip.write_cell(e, z_ptr_read.value(), T::from_bool(cmp)))
        } else {
            WriteRecord::Long(
                memory_chip.write::<NUM_LIMBS>(
                    e,
                    z_ptr_read.value(),
                    z.clone()
                        .into_iter()
                        .map(T::from_canonical_u32)
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap(),
                ),
            )
        };

        let mut x_sign = 0;
        let mut y_sign = 0;

        if opcode == U256Opcode::SLT {
            x_sign = x[NUM_LIMBS - 1] >> (LIMB_BITS - 1);
            y_sign = y[NUM_LIMBS - 1] >> (LIMB_BITS - 1);
            self.xor_lookup_chip
                .request(x[NUM_LIMBS - 1], 1 << (LIMB_BITS - 1));
            self.xor_lookup_chip
                .request(y[NUM_LIMBS - 1], 1 << (LIMB_BITS - 1));
        }

        if ALU_BITWISE_INSTRUCTIONS.contains(&opcode) {
            for i in 0..NUM_LIMBS {
                self.xor_lookup_chip.request(x[i], y[i]);
            }
        } else if opcode != U256Opcode::EQ {
            for z_val in &z {
                self.xor_lookup_chip.request(*z_val, *z_val);
            }
        }

        self.data
            .push(ArithmeticLogicRecord::<T, NUM_LIMBS, LIMB_BITS> {
                from_state,
                instruction: instruction.clone(),
                x_ptr_read,
                y_ptr_read,
                z_ptr_read,
                x_read,
                y_read,
                z_write,
                x_sign: T::from_canonical_u32(x_sign),
                y_sign: T::from_canonical_u32(y_sign),
                cmp_buffer: if ALU_CMP_INSTRUCTIONS.contains(&opcode) {
                    z.into_iter().map(T::from_canonical_u32).collect()
                } else {
                    vec![]
                },
            });

        Ok(ExecutionState {
            pc: from_state.pc + 1,
            timestamp: memory_chip.timestamp().as_canonical_u32() as usize,
        })
    }
}

fn solve_alu<T: PrimeField32, const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    opcode: U256Opcode,
    x: &[u32],
    y: &[u32],
) -> (Vec<u32>, bool) {
    match opcode {
        U256Opcode::ADD => solve_add::<NUM_LIMBS, LIMB_BITS>(x, y),
        U256Opcode::SUB | U256Opcode::LT => solve_subtract::<NUM_LIMBS, LIMB_BITS>(x, y),
        U256Opcode::EQ => solve_eq::<T, NUM_LIMBS, LIMB_BITS>(x, y),
        U256Opcode::XOR => solve_xor::<NUM_LIMBS, LIMB_BITS>(x, y),
        U256Opcode::AND => solve_and::<NUM_LIMBS, LIMB_BITS>(x, y),
        U256Opcode::OR => solve_or::<NUM_LIMBS, LIMB_BITS>(x, y),
        U256Opcode::SLT => {
            let (z, cmp) = solve_subtract::<NUM_LIMBS, LIMB_BITS>(x, y);
            (
                z,
                cmp ^ (x[NUM_LIMBS - 1] >> (LIMB_BITS - 1) != 0)
                    ^ (y[NUM_LIMBS - 1] >> (LIMB_BITS - 1) != 0),
            )
        }
        _ => unreachable!(),
    }
}

fn solve_add<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u32],
    y: &[u32],
) -> (Vec<u32>, bool) {
    let mut z = vec![0u32; NUM_LIMBS];
    let mut carry = vec![0u32; NUM_LIMBS];
    for i in 0..NUM_LIMBS {
        z[i] = x[i] + y[i] + if i > 0 { carry[i - 1] } else { 0 };
        carry[i] = z[i] >> LIMB_BITS;
        z[i] &= (1 << LIMB_BITS) - 1;
    }
    (z, false)
}

fn solve_subtract<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u32],
    y: &[u32],
) -> (Vec<u32>, bool) {
    let mut z = vec![0u32; NUM_LIMBS];
    let mut carry = vec![0u32; NUM_LIMBS];
    for i in 0..NUM_LIMBS {
        let rhs = y[i] + if i > 0 { carry[i - 1] } else { 0 };
        if x[i] >= rhs {
            z[i] = x[i] - rhs;
            carry[i] = 0;
        } else {
            z[i] = x[i] + (1 << LIMB_BITS) - rhs;
            carry[i] = 1;
        }
    }
    (z, carry[NUM_LIMBS - 1] != 0)
}

fn solve_eq<F: PrimeField32, const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u32],
    y: &[u32],
) -> (Vec<u32>, bool) {
    let mut z = vec![0u32; NUM_LIMBS];
    for i in 0..NUM_LIMBS {
        if x[i] != y[i] {
            z[i] = (F::from_canonical_u32(x[i]) - F::from_canonical_u32(y[i]))
                .inverse()
                .as_canonical_u32();
            return (z, false);
        }
    }
    (z, true)
}

fn solve_xor<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u32],
    y: &[u32],
) -> (Vec<u32>, bool) {
    let z = (0..NUM_LIMBS).map(|i| x[i] ^ y[i]).collect();
    (z, false)
}

fn solve_and<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u32],
    y: &[u32],
) -> (Vec<u32>, bool) {
    let z = (0..NUM_LIMBS).map(|i| x[i] & y[i]).collect();
    (z, false)
}

fn solve_or<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u32],
    y: &[u32],
) -> (Vec<u32>, bool) {
    let z = (0..NUM_LIMBS).map(|i| x[i] | y[i]).collect();
    (z, false)
}
