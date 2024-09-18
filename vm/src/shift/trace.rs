use afs_stark_backend::{config::StarkGenericConfig, rap::AnyRap};
use p3_commit::PolynomialSpace;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::Domain;

use super::ShiftChip;
use crate::arch::chips::MachineChip;

// TODO: implement trace generation

impl<F: PrimeField32, const NUM_LIMBS: usize, const LIMB_BITS: usize> MachineChip<F>
    for ShiftChip<F, NUM_LIMBS, LIMB_BITS>
{
    fn generate_trace(self) -> RowMajorMatrix<F> {
        RowMajorMatrix::default(0, 0)
    }

    fn air<SC: StarkGenericConfig>(&self) -> Box<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        Box::new(self.air.clone())
    }

    fn current_trace_height(&self) -> usize {
        0
    }

    fn trace_width(&self) -> usize {
        0
    }
}
