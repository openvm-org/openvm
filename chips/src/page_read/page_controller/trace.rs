use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use p3_uni_stark::StarkGenericConfig;

use super::PageController;

impl<SC: StarkGenericConfig> PageController<SC> {
    /// Every row in the trace is [index] | [mult]
    pub fn generate_trace<F: PrimeField64>(&self) -> RowMajorMatrix<F> {
        RowMajorMatrix::new(
            self.request_count
                .iter()
                .enumerate()
                .flat_map(|(i, c)| {
                    vec![
                        F::from_canonical_usize(i),
                        F::from_canonical_u32(c.load(std::sync::atomic::Ordering::Relaxed)),
                    ]
                })
                .collect(),
            2,
        )
    }
}
