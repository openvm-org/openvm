use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use crate::cpu::trace::ProgramExecution;

use super::ProgramAir;

impl<F: PrimeField64> ProgramAir<F> {
    pub fn generate_trace<const WORD_SIZE: usize>(
        &self,
        execution: &ProgramExecution<WORD_SIZE, F>,
    ) -> RowMajorMatrix<F> {
        let mut frequencies = execution.execution_frequencies.clone();
        while frequencies.len() != self.program.len() {
            frequencies.push(F::zero());
        }
        RowMajorMatrix::new_col(frequencies)
    }
}
