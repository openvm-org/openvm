use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use super::ProgramChip;

impl<F: PrimeField64> ProgramChip<F> {
    pub fn generate_trace(&self) -> RowMajorMatrix<F> {
        RowMajorMatrix::new_col(
            self.execution_frequencies
                .iter()
                .map(|x| F::from_canonical_usize(*x))
                .collect::<Vec<F>>(),
        )
    }
}
