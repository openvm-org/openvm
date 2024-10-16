pub mod air;
pub mod columns;
pub mod trace;

use std::sync::{atomic::AtomicU32, Arc};

use afs_stark_backend::{
    config::StarkGenericConfig, p3_uni_stark::Val, prover::types::AirProofInput, rap::AnyRap, Chip,
    ChipUsageGetter,
};
use air::XorLookupAir;
use p3_field::PrimeField32;

use super::bus::XorBus;
use crate::xor::lookup::columns::NUM_XOR_LOOKUP_COLS;

/// This chip gets requests to compute the xor of two numbers x and y of at most M bits.
/// It generates a preprocessed table with a row for each possible triple (x, y, x^y)
/// and keeps count of the number of times each triple is requested for the single main trace column.
#[derive(Debug)]
pub struct XorLookupChip<const M: usize> {
    pub air: XorLookupAir<M>,
    pub count: Vec<Vec<AtomicU32>>,
}

impl<const M: usize> XorLookupChip<M> {
    pub fn new(bus: XorBus) -> Self {
        let mut count = vec![];
        for _ in 0..(1 << M) {
            let mut row = vec![];
            for _ in 0..(1 << M) {
                row.push(AtomicU32::new(0));
            }
            count.push(row);
        }
        Self {
            air: XorLookupAir::new(bus),
            count,
        }
    }

    /// The xor bus this chip interacts with
    pub fn bus(&self) -> XorBus {
        self.air.bus
    }

    fn calc_xor(&self, x: u32, y: u32) -> u32 {
        x ^ y
    }

    pub fn request(&self, x: u32, y: u32) -> u32 {
        let val_atomic = &self.count[x as usize][y as usize];
        val_atomic.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        self.calc_xor(x, y)
    }

    pub fn clear(&self) {
        for i in 0..(1 << M) {
            for j in 0..(1 << M) {
                self.count[i][j].store(0, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
}

impl<SC: StarkGenericConfig, const M: usize> Chip<SC> for XorLookupChip<M>
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

impl<const M: usize> ChipUsageGetter for XorLookupChip<M> {
    fn air_name(&self) -> String {
        "XorLookupAir".to_string()
    }
    fn current_trace_height(&self) -> usize {
        1 << (2 * M)
    }

    fn trace_width(&self) -> usize {
        NUM_XOR_LOOKUP_COLS
    }
}
