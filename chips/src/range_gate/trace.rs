use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use super::{columns::NUM_RANGE_GATE_COLS, RangeCheckerGateChip};

impl RangeCheckerGateChip {
    pub fn generate_trace<F: PrimeField64>(&self) -> RowMajorMatrix<F> {
        let rows = self
            .count
            .iter()
            .enumerate()
            .flat_map(|(i, count)| {
                let c = count.load(std::sync::atomic::Ordering::Relaxed);
                vec![F::from_canonical_usize(i), F::from_canonical_u32(c)]
            })
            .collect();

        RowMajorMatrix::new(rows, NUM_RANGE_GATE_COLS)
    }
}
