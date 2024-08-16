use afs_primitives::range_gate::RangeCheckerGateChip;
use air::LongAdditionAir;

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[derive(Debug)]
pub struct LongAdditionChip<const ARG_SIZE: usize, const LIMB_SIZE: usize> {
    pub air: LongAdditionAir<ARG_SIZE, LIMB_SIZE>,
    pub range_checker_chip: RangeCheckerGateChip,
    operations: Vec<(Vec<u32>, Vec<u32>)>,
}

impl<const ARG_SIZE: usize, const LIMB_SIZE: usize> LongAdditionChip<ARG_SIZE, LIMB_SIZE> {
    pub fn new(bus_index: usize) -> Self {
        Self {
            air: LongAdditionAir { bus_index },
            range_checker_chip: RangeCheckerGateChip::new(bus_index, 1 << LIMB_SIZE),
            operations: vec![],
        }
    }

    pub fn request(&mut self, operands: Vec<(Vec<u32>, Vec<u32>)>) {
        for (x, y) in operands {
            self.operations.push((x, y));
        }
    }
}
