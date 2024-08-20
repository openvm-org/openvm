use p3_commit::PolynomialSpace;
use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig};

use afs_stark_backend::rap::AnyRap;

use crate::arch::chips::MachineChip;

use super::ProgramChip;

impl<F: PrimeField64> MachineChip<F> for ProgramChip<F> {
    fn generate_trace(&mut self) -> RowMajorMatrix<F> {
        RowMajorMatrix::new_col(
            self.execution_frequencies
                .iter()
                .map(|x| F::from_canonical_usize(*x))
                .collect::<Vec<F>>(),
        )
    }

    fn air<SC: StarkGenericConfig>(&self) -> &dyn AnyRap<SC>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        &self.air
    }
}
