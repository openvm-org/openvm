use std::sync::Arc;

use crate::{is_less_than::IsLessThanAir, range_gate::RangeCheckerGateChip};

pub mod air;
pub mod bridge;
pub mod columns;
#[cfg(test)]
pub mod tests;
pub mod trace;

pub use air::SumAir;

pub struct SumChip {
    pub air: SumAir,
    pub range_checker: Arc<RangeCheckerGateChip>,
}

impl SumChip {
    pub fn new(
        input_bus: usize,
        output_bus: usize,
        key_limb_bits: usize,
        key_decomp: usize,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> Self {
        Self {
            air: SumAir {
                input_bus,
                output_bus,
                is_lt_air: IsLessThanAir::new(range_checker.air.bus, key_limb_bits, key_decomp),
            },
            range_checker,
        }
    }
}
