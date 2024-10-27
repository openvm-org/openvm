use std::{array, sync::Arc};

use air::ShiftCoreAir;
use ax_circuit_primitives::{var_range::VariableRangeCheckerChip, xor::XorLookupChip};
use axvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use p3_field::PrimeField32;

use crate::{
    arch::{
        instructions::{U256Opcode, UsizeOpcode},
        ExecutionBridge, ExecutionBus, ExecutionState, InstructionExecutor,
    },
    system::{
        memory::{MemoryControllerRef, MemoryReadRecord, MemoryWriteRecord},
        program::{ExecutionError, ProgramBus},
    },
};

mod air;
mod bridge;
mod columns;
mod trace;

#[cfg(test)]
mod tests;

#[derive(Clone, Debug)]
pub struct ShiftRecord<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub from_state: ExecutionState<u32>,
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
    pub air: ShiftCoreAir<NUM_LIMBS, LIMB_BITS>,
    data: Vec<ShiftRecord<T, NUM_LIMBS, LIMB_BITS>>,
    memory_controller: MemoryControllerRef<T>,
    pub xor_lookup_chip: Arc<XorLookupChip<LIMB_BITS>>,
    pub range_checker_chip: Arc<VariableRangeCheckerChip>,

    offset: usize,
}

impl<T: PrimeField32, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    ShiftChip<T, NUM_LIMBS, LIMB_BITS>
{
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_controller: MemoryControllerRef<T>,
        xor_lookup_chip: Arc<XorLookupChip<LIMB_BITS>>,
        offset: usize,
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
        let memory_bridge = memory_controller.borrow().memory_bridge();
        let range_checker_chip = memory_controller.borrow().range_checker.clone();
        Self {
            air: ShiftCoreAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
                range_bus: range_checker_chip.bus(),
                xor_bus: xor_lookup_chip.bus(),
                offset,
            },
            data: vec![],
            memory_controller,
            range_checker_chip,
            xor_lookup_chip,
            offset,
        }
    }
}

impl<T: PrimeField32, const NUM_LIMBS: usize, const LIMB_BITS: usize> InstructionExecutor<T>
    for ShiftChip<T, NUM_LIMBS, LIMB_BITS>
{
    fn execute(
        &mut self,
        instruction: Instruction<T>,
        from_state: ExecutionState<u32>,
    ) -> Result<ExecutionState<u32>, ExecutionError> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = instruction.clone();
        let local_opcode_index = opcode - self.offset;
        assert!(U256Opcode::shift_opcodes().any(|op| op as usize == local_opcode_index));

        let mut memory_controller = self.memory_controller.borrow_mut();
        debug_assert_eq!(from_state.timestamp, memory_controller.timestamp());

        let [z_ptr_read, x_ptr_read, y_ptr_read] =
            [a, b, c].map(|ptr_of_ptr| memory_controller.read_cell(d, ptr_of_ptr));
        let x_read = memory_controller.read::<NUM_LIMBS>(e, x_ptr_read.value());
        let y_read = memory_controller.read::<NUM_LIMBS>(e, y_ptr_read.value());

        let x = x_read.data.map(|x| x.as_canonical_u32());
        let y = y_read.data.map(|y| y.as_canonical_u32());
        let (z, limb_shift, bit_shift) =
            run_shift::<NUM_LIMBS, LIMB_BITS>(&x, &y, U256Opcode::from_usize(local_opcode_index));

        let carry = x
            .into_iter()
            .map(
                |val: u32| match U256Opcode::from_usize(local_opcode_index) {
                    U256Opcode::SLL => val >> (LIMB_BITS - bit_shift),
                    _ => val % (1 << bit_shift),
                },
            )
            .collect::<Vec<_>>();

        let mut x_sign = 0;
        if local_opcode_index == U256Opcode::SRA as usize {
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

        let z_write = memory_controller.write::<NUM_LIMBS>(
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
            instruction: Instruction {
                opcode: local_opcode_index,
                ..instruction
            },
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
            pc: from_state.pc + DEFAULT_PC_STEP,
            timestamp: memory_controller.timestamp(),
        })
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        let local_opcode_index = U256Opcode::from_usize(opcode - self.offset);
        format!("{local_opcode_index:?}<{NUM_LIMBS},{LIMB_BITS}>")
    }
}

fn run_shift<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u32],
    y: &[u32],
    op: U256Opcode,
) -> (Vec<u32>, usize, usize) {
    match op {
        U256Opcode::SLL => run_shift_left::<NUM_LIMBS, LIMB_BITS>(x, y),
        U256Opcode::SRL => run_shift_right::<NUM_LIMBS, LIMB_BITS>(x, y, true),
        U256Opcode::SRA => run_shift_right::<NUM_LIMBS, LIMB_BITS>(x, y, false),
        _ => unreachable!(),
    }
}

fn run_shift_left<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
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

fn run_shift_right<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
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
