use afs_primitives::range_gate::RangeCheckerGateChip;
use air::LongArithmeticAir;
use itertools::Itertools;

use crate::cpu::OpCode;

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

pub const fn num_limbs<const ARG_SIZE: usize, const LIMB_SIZE: usize>() -> usize {
    (ARG_SIZE + LIMB_SIZE - 1) / LIMB_SIZE
}

pub struct LongArithmeticOperation {
    pub opcode: OpCode,
    pub operand1: Vec<u32>,
    pub operand2: Vec<u32>,
}

pub struct LongArithmeticChip<const ARG_SIZE: usize, const LIMB_SIZE: usize> {
    pub air: LongArithmeticAir<ARG_SIZE, LIMB_SIZE>,
    pub range_checker_chip: RangeCheckerGateChip,
    operations: Vec<LongArithmeticOperation>,
}

impl<const ARG_SIZE: usize, const LIMB_SIZE: usize> LongArithmeticChip<ARG_SIZE, LIMB_SIZE> {
    pub fn new(bus_index: usize) -> Self {
        Self {
            air: LongArithmeticAir {
                bus_index,
                base_op: OpCode::ADD256,
            },
            range_checker_chip: RangeCheckerGateChip::new(bus_index, 1 << LIMB_SIZE),
            operations: vec![],
        }
    }

    pub fn request(&mut self, ops: Vec<OpCode>, operands: Vec<(Vec<u32>, Vec<u32>)>) {
        for (op, (x, y)) in ops.iter().zip_eq(operands.iter()) {
            // I think that it would be more logical to calculate the result in the
            // trace generation procedure, because, technically, the result is
            // a part of the trace. This means that we need to "prepare" the
            // dependent chips during the trace generation as well, but this
            // seems fine to me.
            self.operations.push(LongArithmeticOperation {
                opcode: *op,
                operand1: x.clone(),
                operand2: y.clone(),
            });
        }
    }
}
