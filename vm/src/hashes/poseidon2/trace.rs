use afs_stark_backend::rap::AnyRap;
use p3_air::BaseAir;
use p3_commit::PolynomialSpace;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig};

use super::{columns::*, Poseidon2Chip};
use crate::arch::chips::MachineChip;

impl<const WIDTH: usize, F: PrimeField32> MachineChip<F> for Poseidon2Chip<WIDTH, F> {
    /// Generates final Poseidon2VmAir trace from cached rows.
    fn generate_trace(self) -> RowMajorMatrix<F> {
        let row_len = self.rows.len();
        let correct_len = row_len.next_power_of_two();
        let diff = correct_len - row_len;

        let mut flat_rows = Vec::with_capacity(correct_len * self.air.width());
        for row in self.rows.iter() {
            flat_rows.extend(row.flatten());
        }
        for _ in 0..diff {
            flat_rows.extend(self.blank_row().flatten());
        }

        RowMajorMatrix::new(flat_rows, self.air.width())
    }

    fn air<SC: StarkGenericConfig>(&self) -> Box<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        Box::new(self.air.clone())
    }

    fn current_trace_height(&self) -> usize {
        self.rows.len()
    }

    fn trace_width(&self) -> usize {
        self.air.width()
    }
}

impl<const WIDTH: usize, F: PrimeField32> Poseidon2Chip<WIDTH, F> {
    fn blank_row(&self) -> Poseidon2VmCols<WIDTH, F> {
        Poseidon2VmCols::<WIDTH, F>::blank_row(&self.air)
    }
}
