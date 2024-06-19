use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use crate::cpu::trace::ProgramExecution;

use super::ProgramAir;

impl<F: PrimeField64> ProgramAir<F> {
    pub fn generate_trace(&self, execution: &ProgramExecution<F>) -> RowMajorMatrix<F> {
        RowMajorMatrix::new_col(execution.execution_frequencies.clone())
    }
}
