use afs_stark_backend::rap::AnyRap;
use p3_air::BaseAir;
use p3_commit::PolynomialSpace;
use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig};

use super::ProgramChip;
use crate::arch::chips::SingleAirMachineChip;

impl<F: PrimeField64> SingleAirMachineChip<F> for ProgramChip<F> {
    fn generate_trace(self) -> RowMajorMatrix<F> {
        RowMajorMatrix::new_col(
            self.execution_frequencies
                .iter()
                .map(|x| F::from_canonical_usize(*x))
                .collect::<Vec<F>>(),
        )
    }

    fn air<SC: StarkGenericConfig>(&self) -> Box<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        Box::new(self.air.clone())
    }

    fn current_trace_height(&self) -> usize {
        self.true_program_length
    }

    fn trace_width(&self) -> usize {
        self.air.width()
    }
}
