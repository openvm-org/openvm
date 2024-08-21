use afs_primitives::range_gate::RangeCheckerGateChip;
use air::LongMultiplicationAir;

use crate::cpu::OpCode;

#[cfg(test)]
mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

pub struct LongMultiplicationChip {
    pub arg_size: usize,
    pub limb_size: usize,
    pub mul_op: OpCode,
    pub air: LongMultiplicationAir,
    pub range_checker_chip: RangeCheckerGateChip,
    operations: Vec<LongMultiplicationOperation>,
}

pub struct LongMultiplicationOperation {
    pub opcode: OpCode, // always MUL, but we need to check it in the AIR, so
    pub multiplicand: Vec<u32>,
    pub multiplier: Vec<u32>,
}

pub fn num_limbs(arg_size: usize, limb_size: usize) -> usize {
    (arg_size + limb_size - 1) / limb_size
}

impl LongMultiplicationChip {
    pub fn new(bus_index: usize, arg_size: usize, limb_size: usize, mul_op: OpCode) -> Self {
        Self {
            arg_size,
            limb_size,
            mul_op,
            air: LongMultiplicationAir {
                arg_size,
                limb_size,
                bus_index,
                mul_op,
            },
            range_checker_chip: RangeCheckerGateChip::new(
                bus_index,
                (num_limbs(arg_size, limb_size) as u32) << limb_size,
            ),
            operations: vec![],
        }
    }

    pub fn request(&mut self, ops: Vec<OpCode>, operands: Vec<(Vec<u32>, Vec<u32>)>) {
        for (op, (x, y)) in ops.into_iter().zip(operands) {
            self.operations.push(LongMultiplicationOperation {
                opcode: op,
                multiplicand: x,
                multiplier: y,
            });
        }
    }
}
