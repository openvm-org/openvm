use std::{array, sync::Arc};

use afs_primitives::{var_range::VariableRangeCheckerChip, xor::lookup::XorLookupChip};
use air::ShiftAir;
use p3_field::PrimeField32;

use crate::{
    arch::{
        bridge::ExecutionBridge,
        bus::ExecutionBus,
        chips::InstructionExecutor,
        columns::ExecutionState,
        instructions::{Opcode, SHIFT_256_INSTRUCTIONS},
    },
    memory::{MemoryChipRef, MemoryReadRecord, MemoryWriteRecord},
    program::{bridge::ProgramBus, ExecutionError, Instruction},
};

mod air;
mod bridge;
mod columns;
mod trace;

#[cfg(test)]
mod tests;

#[derive(Clone, Debug)]
pub struct ShiftRecord<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub from_state: ExecutionState<usize>,
    pub instruction: Instruction<T>,
    pub x_ptr_read: MemoryReadRecord<T, 1>,
    pub y_ptr_read: MemoryReadRecord<T, 1>,
    pub z_ptr_read: MemoryReadRecord<T, 1>,
    pub x_read: MemoryReadRecord<T, NUM_LIMBS>,
    pub y_read: MemoryReadRecord<T, NUM_LIMBS>,
    pub z_write: MemoryWriteRecord<T, NUM_LIMBS>,
    pub bit_shift_carry: [T; NUM_LIMBS],
    pub bit_shift: usize,
    pub limb_shift: usize,
    pub x_sign: T,
}

#[derive(Clone, Debug)]
pub struct ShiftChip<T: PrimeField32, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub air: ShiftAir<NUM_LIMBS, LIMB_BITS>,
    data: Vec<ShiftRecord<T, NUM_LIMBS, LIMB_BITS>>,
    memory_chip: MemoryChipRef<T>,
    pub xor_lookup_chip: Arc<XorLookupChip<LIMB_BITS>>,
    pub range_checker_chip: Arc<VariableRangeCheckerChip>,
}

impl<T: PrimeField32, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    ShiftChip<T, NUM_LIMBS, LIMB_BITS>
{
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_chip: MemoryChipRef<T>,
        xor_lookup_chip: Arc<XorLookupChip<LIMB_BITS>>,
    ) -> Self {
        // (1 << (2 * LIMB_BITS)) fits within a u32
        assert!(LIMB_BITS < 16, "LIMB_BITS {} >= 16", LIMB_BITS);
        // For range check that bit_shift < LIMB_BITS
        assert!(
            LIMB_BITS.is_power_of_two(),
            "LIMB_BITS {} not a power of 2",
            LIMB_BITS
        );
        // A non-overflow shift amount is defined entirely within y[0] and y[1]
        assert!(
            NUM_LIMBS * LIMB_BITS < (1 << (2 * LIMB_BITS)),
            "NUM_LIMBS * LIMB_BITS {} >= 2^(2 * LIMB_BITS {})",
            NUM_LIMBS * LIMB_BITS,
            LIMB_BITS
        );
        let memory_bridge = memory_chip.borrow().memory_bridge();
        let range_checker_chip = memory_chip.borrow().range_checker.clone();
        Self {
            air: ShiftAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
                range_bus: range_checker_chip.bus(),
                xor_bus: xor_lookup_chip.bus(),
            },
            data: vec![],
            memory_chip,
            range_checker_chip,
            xor_lookup_chip,
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
        assert!(SHIFT_256_INSTRUCTIONS.contains(&opcode));

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
        let (z, limb_shift, bit_shift) = solve_shift::<NUM_LIMBS, LIMB_BITS>(&x, &y, opcode);

        let carry = x
            .into_iter()
            .map(|val: u32| match opcode {
                Opcode::SLL256 => val >> (LIMB_BITS - bit_shift),
                _ => val % (1 << bit_shift),
            })
            .collect::<Vec<_>>();

        let mut x_sign = 0;
        if opcode == Opcode::SRA256 {
            x_sign = x[NUM_LIMBS - 1] >> (LIMB_BITS - 1);
            self.xor_lookup_chip
                .request(x[NUM_LIMBS - 1], 1 << (LIMB_BITS - 1));
        }

        self.range_checker_chip
            .add_count(bit_shift as u32, LIMB_BITS.ilog2() as usize);
        for (z_val, carry_val) in z.iter().zip(carry.iter()) {
            self.range_checker_chip.add_count(*z_val, LIMB_BITS);
            self.range_checker_chip.add_count(*carry_val, bit_shift);
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

        self.data.push(ShiftRecord {
            from_state,
            instruction: instruction.clone(),
            x_ptr_read,
            y_ptr_read,
            z_ptr_read,
            x_read,
            y_read,
            z_write,
            bit_shift_carry: array::from_fn(|i| T::from_canonical_u32(carry[i])),
            bit_shift,
            limb_shift,
            x_sign: T::from_canonical_u32(x_sign),
        });

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
) -> (Vec<u32>, usize, usize) {
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
) -> (Vec<u32>, usize, usize) {
    let mut result = vec![0u32; NUM_LIMBS];

    let (is_zero, limb_shift, bit_shift) = get_shift::<NUM_LIMBS, LIMB_BITS>(y);
    if is_zero {
        return (result, limb_shift, bit_shift);
    }

    for i in limb_shift..NUM_LIMBS {
        result[i] = if i > limb_shift {
            ((x[i - limb_shift] << bit_shift) + (x[i - limb_shift - 1] >> (LIMB_BITS - bit_shift)))
                % (1 << LIMB_BITS)
        } else {
            (x[i - limb_shift] << bit_shift) % (1 << LIMB_BITS)
        };
    }
    (result, limb_shift, bit_shift)
}

fn solve_shift_right<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u32],
    y: &[u32],
    logical: bool,
) -> (Vec<u32>, usize, usize) {
    let fill = if logical {
        0
    } else {
        ((1 << LIMB_BITS) - 1) * (x[NUM_LIMBS - 1] >> (LIMB_BITS - 1))
    };
    let mut result = vec![fill; NUM_LIMBS];

    let (is_zero, limb_shift, bit_shift) = get_shift::<NUM_LIMBS, LIMB_BITS>(y);
    if is_zero {
        return (result, limb_shift, bit_shift);
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
    (result, limb_shift, bit_shift)
}

fn get_shift<const NUM_LIMBS: usize, const LIMB_BITS: usize>(y: &[u32]) -> (bool, usize, usize) {
    // We assume `NUM_LIMBS * LIMB_BITS < 2^(2*LIMB_BITS)` so if there are any higher limbs,
    // the shifted value is zero.
    let shift = (y[0] + (y[1] * (1 << LIMB_BITS))) as usize;
    if shift < NUM_LIMBS * LIMB_BITS && y[2..].iter().all(|&val| val == 0) {
        (false, shift / LIMB_BITS, shift % LIMB_BITS)
    } else {
        (true, NUM_LIMBS, shift % LIMB_BITS)
    }
}
