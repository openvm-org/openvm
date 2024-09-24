use air::ShiftAir;
use p3_field::PrimeField32;

use crate::{
    arch::{
        bus::ExecutionBus,
        chips::InstructionExecutor,
        columns::ExecutionState,
        instructions::{Opcode, ALU_256_INSTRUCTIONS},
    },
    memory::MemoryChipRef,
    program::{ExecutionError, Instruction},
};

mod air;
mod trace;

#[cfg(test)]
mod tests;

#[derive(Debug)]
pub struct ShiftChip<T: PrimeField32, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub air: ShiftAir<NUM_LIMBS, LIMB_BITS>,
    // TODO: add data storage for trace generation
    memory_chip: MemoryChipRef<T>,
    // TODO: add range checker
}

impl<T: PrimeField32, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    ShiftChip<T, NUM_LIMBS, LIMB_BITS>
{
    pub fn new(execution_bus: ExecutionBus, memory_chip: MemoryChipRef<T>) -> Self {
        assert!(LIMB_BITS < 16, "LIMB_BITS {} >= 16", LIMB_BITS);
        assert!(
            NUM_LIMBS < (1 << (2 * LIMB_BITS)),
            "NUM_LIMBS {} >= 2^(2 * LIMB_BITS {})",
            NUM_LIMBS,
            LIMB_BITS
        );
        let memory_bridge = memory_chip.borrow().memory_bridge();
        Self {
            air: ShiftAir {
                execution_bus,
                memory_bridge,
            },
            memory_chip,
        }
    }
}

impl<T: PrimeField32, const NUM_LIMBS: usize, const LIMB_BITS: usize> InstructionExecutor<T>
    for ShiftChip<T, NUM_LIMBS, LIMB_BITS>
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
        assert!(ALU_256_INSTRUCTIONS.contains(&opcode));

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
        let y = y_read.data.map(|y| y.as_canonical_u32());
        let z = solve_shift::<NUM_LIMBS, LIMB_BITS>(&x, &y, opcode);

        // TODO: range check add count

        let _z_write = memory_chip.write::<NUM_LIMBS>(
            e,
            z_ptr_read.value(),
            z.into_iter()
                .map(T::from_canonical_u32)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        );

        // TODO: push information to data for trace generation

        Ok(ExecutionState {
            pc: from_state.pc + 1,
            timestamp: memory_chip.timestamp().as_canonical_u32() as usize,
        })
    }
}

fn solve_shift<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u32],
    y: &[u32],
    op: Opcode,
) -> Vec<u32> {
    match op {
        Opcode::SLL256 => solve_shift_left::<NUM_LIMBS, LIMB_BITS>(x, y),
        Opcode::SRL256 => solve_shift_right::<NUM_LIMBS, LIMB_BITS>(x, y, true),
        Opcode::SRA256 => solve_shift_right::<NUM_LIMBS, LIMB_BITS>(x, y, false),
        _ => unreachable!(),
    }
}

fn solve_shift_left<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u32],
    y: &[u32],
) -> Vec<u32> {
    let mut result = vec![0u32; NUM_LIMBS];

    let (is_zero, limb_shift, bit_shift) = get_shift::<NUM_LIMBS, LIMB_BITS>(y);
    if is_zero {
        return result;
    }

    for i in limb_shift..NUM_LIMBS {
        result[i] = if i > limb_shift {
            ((x[i - limb_shift] << bit_shift) + (x[i - limb_shift - 1] >> (LIMB_BITS - bit_shift)))
                % (1 << LIMB_BITS)
        } else {
            (x[i - limb_shift] << bit_shift) % (1 << LIMB_BITS)
        };
    }
    result
}

fn solve_shift_right<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u32],
    y: &[u32],
    logical: bool,
) -> Vec<u32> {
    let fill =
        (1 - (logical as u32)) * ((1 << LIMB_BITS) - 1) * (x[NUM_LIMBS - 1] >> (LIMB_BITS - 1));
    let mut result = vec![fill; NUM_LIMBS];

    let (is_zero, limb_shift, bit_shift) = get_shift::<NUM_LIMBS, LIMB_BITS>(y);
    if is_zero {
        return result;
    }

    for i in 0..(NUM_LIMBS - limb_shift) {
        result[i] = if i + limb_shift + 1 < NUM_LIMBS {
            ((x[i + limb_shift] >> bit_shift) + (x[i + limb_shift + 1] << (LIMB_BITS - bit_shift)))
                % (1 << LIMB_BITS)
        } else {
            ((x[i + limb_shift] >> bit_shift) + (fill << (LIMB_BITS - bit_shift)))
                % (1 << LIMB_BITS)
        }
    }
    result
}

fn get_shift<const NUM_LIMBS: usize, const LIMB_BITS: usize>(y: &[u32]) -> (bool, usize, usize) {
    // We assume `NUM_LIMBS < 2^(2*LIMB_BITS)` so if there are any higher limbs, the shifted value is zero.
    if y[2..].iter().any(|&val| val != 0) {
        return (true, 0, 0);
    }
    let shift = (y[0] + (y[1] * (1 << LIMB_BITS))) as usize;
    if shift < NUM_LIMBS * LIMB_BITS {
        (false, shift / LIMB_BITS, shift % LIMB_BITS)
    } else {
        (true, 0, 0)
    }
}
