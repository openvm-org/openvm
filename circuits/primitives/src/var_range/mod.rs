use std::sync::{atomic::AtomicU32, Arc};

use afs_stark_backend::p3_uni_stark::Val;
use p3_field::PrimeField32;

pub mod air;
pub mod bus;
pub mod columns;
pub mod trace;

#[cfg(test)]
pub mod tests;

use afs_stark_backend::{
    config::StarkGenericConfig, prover::types::AirProofInput, rap::AnyRap, Chip, ChipUsageGetter,
};
pub use air::VariableRangeCheckerAir;
use bus::VariableRangeCheckerBus;
use columns::NUM_VARIABLE_RANGE_COLS;

#[derive(Debug)]
pub struct VariableRangeCheckerChip {
    pub air: VariableRangeCheckerAir,
    count: Vec<AtomicU32>,
}

impl VariableRangeCheckerChip {
    pub fn new(bus: VariableRangeCheckerBus) -> Self {
        let num_rows = (1 << (bus.range_max_bits + 1)) as usize;
        let count = (0..num_rows).map(|_| AtomicU32::new(0)).collect();
        Self {
            air: VariableRangeCheckerAir::new(bus),
            count,
        }
    }

    pub fn bus(&self) -> VariableRangeCheckerBus {
        self.air.bus
    }

    pub fn range_max_bits(&self) -> usize {
        self.air.range_max_bits()
    }

    pub fn air_width(&self) -> usize {
        NUM_VARIABLE_RANGE_COLS
    }

    pub fn add_count(&self, value: u32, max_bits: usize) {
        // index is 2^max_bits + value - 1 + 1 for the extra [0, 0] row
        // if each [value, max_bits] is valid, the sends multiset will be exactly the receives multiset
        let idx = (1 << max_bits) + (value as usize);
        assert!(
            idx < self.count.len(),
            "range exceeded: {} >= {}",
            idx,
            self.count.len()
        );
        let val_atomic = &self.count[idx];
        val_atomic.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn clear(&self) {
        for i in 0..self.count.len() {
            self.count[i].store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

impl<SC: StarkGenericConfig> Chip<SC> for VariableRangeCheckerChip
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        Arc::new(self.air)
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        let trace = self.generate_trace::<Val<SC>>();
        AirProofInput::simple_no_pis(Arc::new(self.air), trace)
    }
}

impl ChipUsageGetter for VariableRangeCheckerChip {
    fn air_name(&self) -> String {
        "VariableRangeCheckerAir".to_string()
    }
    fn current_trace_height(&self) -> usize {
        self.count.len()
    }

    fn trace_width(&self) -> usize {
        NUM_VARIABLE_RANGE_COLS
    }
}
