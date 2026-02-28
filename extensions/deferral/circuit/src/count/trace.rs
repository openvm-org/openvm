use std::{
    borrow::BorrowMut,
    sync::atomic::{AtomicU32, Ordering},
};

use openvm_circuit::utils::next_power_of_two_or_zero;
use openvm_circuit_primitives::Chip;
use openvm_stark_backend::{
    p3_field::PrimeCharacteristicRing,
    p3_matrix::dense::RowMajorMatrix,
    prover::{AirProvingContext, ColMajorMatrix, CpuBackend},
    StarkProtocolConfig, Val,
};

use crate::count::DeferralCircuitCountCols;

#[derive(Debug)]
pub struct DeferralCircuitCountChip {
    pub count: Vec<AtomicU32>,
}

impl DeferralCircuitCountChip {
    pub fn new(num_deferral_circuit: usize) -> Self {
        let count = (0..num_deferral_circuit)
            .map(|_| AtomicU32::new(0))
            .collect();
        Self { count }
    }

    pub fn add_count(&self, idx: u32) {
        let idx = idx as usize;
        assert!(idx < self.count.len());
        let val_atomic = &self.count[idx];
        val_atomic.fetch_add(1, Ordering::Relaxed);
    }
}

impl<SC: StarkProtocolConfig, RA> Chip<RA, CpuBackend<SC>> for DeferralCircuitCountChip
where
    Val<SC>: PrimeCharacteristicRing,
{
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<CpuBackend<SC>> {
        let width = DeferralCircuitCountCols::<u8>::width();
        let height = next_power_of_two_or_zero(self.count.len());
        let mut trace = vec![Val::<SC>::ZERO; width * height];

        let mut rows = trace.chunks_exact_mut(width);
        let mut row_idx = 0u32;

        for mult in &self.count {
            let row = rows.next().unwrap();
            let cols: &mut DeferralCircuitCountCols<Val<SC>> = (*row).borrow_mut();
            cols.is_valid = Val::<SC>::ONE;
            cols.row_idx = Val::<SC>::from_u32(row_idx);
            cols.mult = Val::<SC>::from_u32(mult.swap(0, Ordering::Relaxed));
            row_idx += 1;
        }

        for row in rows {
            let cols: &mut DeferralCircuitCountCols<Val<SC>> = (*row).borrow_mut();
            cols.row_idx = Val::<SC>::from_u32(row_idx);
            row_idx += 1;
        }

        let trace = RowMajorMatrix::new(trace, width);
        AirProvingContext::simple_no_pis(ColMajorMatrix::from_row_major(&trace))
    }
}
