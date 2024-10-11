use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use super::RangeTupleCheckerChip;

impl<const N: usize> RangeTupleCheckerChip<N> {
    pub fn generate_trace<F: PrimeField32>(&self) -> RowMajorMatrix<F> {
        let rows = self
            .count
            .iter()
            .map(|c| F::from_canonical_u32(c.load(std::sync::atomic::Ordering::SeqCst)))
            .collect::<Vec<_>>();
        RowMajorMatrix::new(rows, 1)
    }
}
