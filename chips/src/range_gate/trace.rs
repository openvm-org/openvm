use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use super::{columns::NUM_RANGE_GATE_COLS, RangeCheckerGateChip};

impl<const MAX: u32> RangeCheckerGateChip<MAX> {
    pub fn generate_trace<F: PrimeField64>(&self) -> RowMajorMatrix<F> {
        let rows = self
            .count
            .iter()
            .enumerate()
            .map(|(i, count)| {
                let c = count.load(std::sync::atomic::Ordering::Relaxed);
                vec![F::from_canonical_usize(i), F::from_canonical_u32(c)]
            })
            .collect::<Vec<_>>();

        RowMajorMatrix::new(rows.concat(), NUM_RANGE_GATE_COLS)
    }
}
