use std::sync::Arc;

use afs_primitives::xor::lookup::XorLookupChip;
use air::UintArithmeticAir;
use p3_field::PrimeField32;

use crate::{
    arch::{
        bridge::ExecutionBridge,
        bus::ExecutionBus,
        chips::InstructionExecutor,
        columns::ExecutionState,
        instructions::{Opcode, UINT256_ARITHMETIC_INSTRUCTIONS},
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

pub const ALU_CMP_INSTRUCTIONS: [Opcode; 3] = [Opcode::LT256, Opcode::EQ256, Opcode::SLT256];
pub const ALU_ARITHMETIC_INSTRUCTIONS: [Opcode; 2] = [Opcode::ADD256, Opcode::SUB256];
pub const ALU_BITWISE_INSTRUCTIONS: [Opcode; 3] = [Opcode::XOR256, Opcode::AND256, Opcode::OR256];

#[derive(Debug)]
pub enum WriteRecord<T, const NUM_LIMBS: usize> {
    Uint(MemoryWriteRecord<T, NUM_LIMBS>),
    Short(MemoryWriteRecord<T, 1>),
}

#[derive(Debug)]
pub struct UintArithmeticRecord<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub from_state: ExecutionState<usize>,
    pub instruction: Instruction<T>,

    pub x_ptr_read: MemoryReadRecord<T, 1>,
    pub y_ptr_read: MemoryReadRecord<T, 1>,
    pub z_ptr_read: MemoryReadRecord<T, 1>,

    pub x_read: MemoryReadRecord<T, NUM_LIMBS>,
    pub y_read: MemoryReadRecord<T, NUM_LIMBS>,
    pub z_write: WriteRecord<T, NUM_LIMBS>,

    // least significant LIMB_BITS - 1 digits of the most significant limbs of x and y
    // if SLT, else should be equal to the most significant limb of x and y
    pub x_msb_masked: T,
    pub y_msb_masked: T,

    // empty if not bool instruction, else contents of this vector will be stored in z
    pub cmp_buffer: Vec<T>,
}

#[derive(Debug)]
pub struct UintArithmeticChip<T: PrimeField32, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub air: UintArithmeticAir<NUM_LIMBS, LIMB_BITS>,
    data: Vec<UintArithmeticRecord<T, NUM_LIMBS, LIMB_BITS>>,
    memory_chip: MemoryChipRef<T>,
    pub xor_lookup_chip: Arc<XorLookupChip<LIMB_BITS>>,
}

impl<T: PrimeField32, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    UintArithmeticChip<T, NUM_LIMBS, LIMB_BITS>
{
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_chip: MemoryChipRef<T>,
        xor_lookup_chip: Arc<XorLookupChip<LIMB_BITS>>,
    ) -> Self {
        let memory_bridge = memory_chip.borrow().memory_bridge();
        Self {
            air: UintArithmeticAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
                bus: xor_lookup_chip.bus(),
            },
            data: vec![],
            memory_chip,
            xor_lookup_chip,
        }
    }
}

impl<T: PrimeField32, const NUM_LIMBS: usize, const LIMB_BITS: usize> InstructionExecutor<T>
    for UintArithmeticChip<T, NUM_LIMBS, LIMB_BITS>
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
        assert!(UINT256_ARITHMETIC_INSTRUCTIONS.contains(&opcode));

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
            WriteRecord::Short(memory_chip.write_cell(e, z_ptr_read.value(), T::from_bool(cmp)))
        } else {
            WriteRecord::Uint(
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

        let slt_mask = if opcode == Opcode::SLT256 {
            (1 << (LIMB_BITS - 1)) - 1
        } else {
            (1 << LIMB_BITS) - 1
        };
        let x_msb_masked = T::from_canonical_u32(x[NUM_LIMBS - 1] & slt_mask);
        let y_msb_masked = T::from_canonical_u32(y[NUM_LIMBS - 1] & slt_mask);

        if ALU_BITWISE_INSTRUCTIONS.contains(&opcode) {
            for i in 0..NUM_LIMBS {
                self.xor_lookup_chip.request(x[i], y[i]);
            }
        } else if opcode != Opcode::EQ256 {
            for z_val in &z {
                self.xor_lookup_chip.request(*z_val, *z_val);
            }
        }

        self.data
            .push(UintArithmeticRecord::<T, NUM_LIMBS, LIMB_BITS> {
                from_state,
                instruction: instruction.clone(),
                x_ptr_read,
                y_ptr_read,
                z_ptr_read,
                x_read,
                y_read,
                z_write,
                x_msb_masked,
                y_msb_masked,
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
    opcode: Opcode,
    x: &[u32],
    y: &[u32],
) -> (Vec<u32>, bool) {
    match opcode {
        Opcode::ADD256 => solve_add::<NUM_LIMBS, LIMB_BITS>(x, y),
        Opcode::SUB256 => solve_subtract::<NUM_LIMBS, LIMB_BITS>(x, y),
        Opcode::LT256 => solve_subtract::<NUM_LIMBS, LIMB_BITS>(x, y),
        Opcode::EQ256 => solve_eq::<T, NUM_LIMBS, LIMB_BITS>(x, y),
        Opcode::XOR256 => solve_xor::<NUM_LIMBS, LIMB_BITS>(x, y),
        Opcode::AND256 => solve_and::<NUM_LIMBS, LIMB_BITS>(x, y),
        Opcode::OR256 => solve_or::<NUM_LIMBS, LIMB_BITS>(x, y),
        Opcode::SLT256 => {
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
