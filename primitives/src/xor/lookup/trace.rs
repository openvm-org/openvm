use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use super::XorLookupChip;

impl<const M: usize> XorLookupChip<M> {
    pub fn generate_trace<F: PrimeField64>(&self) -> RowMajorMatrix<F> {
        let mut counts = vec![];
        for x in 0..(1 << M) {
            for y in 0..(1 << M) {
                counts.push(F::from_canonical_u32(
                    self.count[x][y].load(std::sync::atomic::Ordering::SeqCst),
                ));
            }
        }

        RowMajorMatrix::new_col(counts)
    }
}
