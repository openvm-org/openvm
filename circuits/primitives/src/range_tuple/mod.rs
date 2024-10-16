// Adapted from Valida

use std::sync::{atomic::AtomicU32, Arc};

use afs_stark_backend::p3_uni_stark::Val;
use p3_field::PrimeField32;

mod air;
mod bus;
mod columns;
mod trace;

#[cfg(test)]
pub mod tests;

use afs_stark_backend::{
    config::StarkGenericConfig, prover::types::AirProofInput, rap::AnyRap, Chip, ChipUsageGetter,
};
pub use air::*;
pub use bus::*;

use crate::range_tuple::columns::NUM_RANGE_TUPLE_COLS;

#[derive(Clone, Debug)]
pub struct RangeTupleCheckerChip<const N: usize> {
    pub air: RangeTupleCheckerAir<N>,
    count: Vec<Arc<AtomicU32>>,
}

impl<const N: usize> RangeTupleCheckerChip<N> {
    pub fn new(bus: RangeTupleCheckerBus<N>) -> Self {
        let range_max = bus.sizes.iter().product();
        let count = (0..range_max)
            .map(|_| Arc::new(AtomicU32::new(0)))
            .collect();

        Self {
            air: RangeTupleCheckerAir { bus },
            count,
        }
    }

    pub fn bus(&self) -> &RangeTupleCheckerBus<N> {
        &self.air.bus
    }

    pub fn sizes(&self) -> &[u32; N] {
        &self.air.bus.sizes
    }

    pub fn add_count(&self, ids: &[u32]) {
        let index = ids
            .iter()
            .zip(self.air.bus.sizes.iter())
            .fold(0, |acc, (id, sz)| acc * sz + id) as usize;
        assert!(
            index < self.count.len(),
            "range exceeded: {} >= {}",
            index,
            self.count.len()
        );
        let val_atomic = &self.count[index];
        val_atomic.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

impl<SC: StarkGenericConfig, const N: usize> Chip<SC> for RangeTupleCheckerChip<N>
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

impl<const N: usize> ChipUsageGetter for RangeTupleCheckerChip<N> {
    fn air_name(&self) -> String {
        "RangeTupleCheckerAir".to_string()
    }
    fn current_trace_height(&self) -> usize {
        self.count.len()
    }

    fn trace_width(&self) -> usize {
        NUM_RANGE_TUPLE_COLS
    }
}
