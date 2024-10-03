use std::sync::Arc;

use afs_primitives::range_tuple::RangeTupleCheckerChip;
use p3_field::PrimeField32;

use crate::{
    arch::{
        instructions::U256Opcode, ExecutionBridge, ExecutionBus, ExecutionState,
        InstructionExecutor,
    },
    memory::{MemoryChipRef, MemoryReadRecord, MemoryWriteRecord},
    program::{bridge::ProgramBus, ExecutionError, Instruction},
};

mod air;
mod bridge;
mod columns;
mod trace;

pub use air::*;
pub use columns::*;

#[cfg(test)]
pub mod tests;

#[derive(Clone, Debug)]
pub struct UintMultiplicationRecord<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub from_state: ExecutionState<usize>,
    pub instruction: Instruction<T>,
    pub x_ptr_read: MemoryReadRecord<T, 1>,
    pub y_ptr_read: MemoryReadRecord<T, 1>,
    pub z_ptr_read: MemoryReadRecord<T, 1>,
    pub x_read: MemoryReadRecord<T, NUM_LIMBS>,
    pub y_read: MemoryReadRecord<T, NUM_LIMBS>,
    pub z_write: MemoryWriteRecord<T, NUM_LIMBS>,
    pub carry: Vec<T>,
}

#[derive(Clone, Debug)]
pub struct UintMultiplicationChip<T: PrimeField32, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub air: UintMultiplicationAir<NUM_LIMBS, LIMB_BITS>,
    data: Vec<UintMultiplicationRecord<T, NUM_LIMBS, LIMB_BITS>>,
    memory_chip: MemoryChipRef<T>,
    pub range_tuple_chip: Arc<RangeTupleCheckerChip>,

    offset: usize,
}

impl<T: PrimeField32, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    UintMultiplicationChip<T, NUM_LIMBS, LIMB_BITS>
{
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_chip: MemoryChipRef<T>,
        range_tuple_chip: Arc<RangeTupleCheckerChip>,
        offset: usize,
    ) -> Self {
        assert!(LIMB_BITS < 16, "LIMB_BITS {} >= 16", LIMB_BITS);

        let bus = range_tuple_chip.bus();

        assert_eq!(bus.sizes.len(), 2);
        assert!(
            bus.sizes[0] >= 1 << LIMB_BITS,
            "bus.sizes[0] {} < 2^LIMB_BITS {}",
            bus.sizes[0],
            1 << LIMB_BITS
        );
        assert!(
            bus.sizes[1] >= (NUM_LIMBS * (1 << LIMB_BITS)) as u32,
            "bus.sizes[1] {} < (NUM_LIMBS * 2^LIMB_BITS) {}",
            bus.sizes[1],
            NUM_LIMBS * (1 << LIMB_BITS)
        );

        let memory_bridge = memory_chip.borrow().memory_bridge();
        Self {
            air: UintMultiplicationAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
                bus: bus.clone(),
                offset,
            },
            data: vec![],
            memory_chip,
            range_tuple_chip,
            offset,
        }
    }
}

impl<T: PrimeField32, const NUM_LIMBS: usize, const LIMB_BITS: usize> InstructionExecutor<T>
    for UintMultiplicationChip<T, NUM_LIMBS, LIMB_BITS>
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
        } = instruction;
        let opcode = opcode - self.offset;
        assert!(opcode == U256Opcode::MUL as usize);

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
        let (z, carry) = solve_uint_multiplication::<NUM_LIMBS, LIMB_BITS>(&x, &y);

        for (z_val, carry_val) in z.iter().zip(carry.iter()) {
            self.range_tuple_chip.add_count(&[*z_val, *carry_val]);
        }

        let z_write = memory_chip.write::<NUM_LIMBS>(
            e,
            z_ptr_read.value(),
            z.into_iter()
                .map(T::from_canonical_u32)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        );

        self.data.push(UintMultiplicationRecord {
            from_state,
            instruction: instruction.clone(),
            x_ptr_read,
            y_ptr_read,
            z_ptr_read,
            x_read,
            y_read,
            z_write,
            carry: carry.into_iter().map(T::from_canonical_u32).collect(),
        });

        Ok(ExecutionState {
            pc: from_state.pc + 1,
            timestamp: memory_chip.timestamp().as_canonical_u32() as usize,
        })
    }
}

fn solve_uint_multiplication<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u32],
    y: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    let mut result = vec![0; NUM_LIMBS];
    let mut carry = vec![0; NUM_LIMBS];
    for i in 0..NUM_LIMBS {
        if i > 0 {
            result[i] = carry[i - 1];
        }
        for j in 0..=i {
            result[i] += x[j] * y[i - j];
        }
        carry[i] = result[i] >> LIMB_BITS;
        result[i] %= 1 << LIMB_BITS;
    }
    (result, carry)
}
