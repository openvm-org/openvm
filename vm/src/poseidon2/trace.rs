use p3_air::BaseAir;
use p3_commit::PolynomialSpace;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig};

use afs_primitives::sub_chip::LocalTraceInstructions;
use afs_stark_backend::rap::AnyRap;

use crate::{
    arch::{chips::MachineChip, columns::ExecutionState},
    cpu::trace::Instruction,
};

use super::{columns::*, Poseidon2Chip, Poseidon2VmAir};

impl<const WIDTH: usize, F: PrimeField32> Poseidon2VmAir<WIDTH, F> {
    /// Generates a single row from inputs.
    pub fn generate_row(
        &self,
        execution_state: ExecutionState<usize>,
        instruction: &Instruction<F>,
        dst: F,
        lhs: F,
        rhs: F,
        input_state: [F; WIDTH],
    ) -> Poseidon2VmCols<WIDTH, F> {
        // SAFETY: only allowed because WIDTH constrained to 16 above
        let internal = self.inner.generate_trace_row(input_state);
        Poseidon2VmCols {
            io: Poseidon2VmAir::<WIDTH, F>::make_io_cols(execution_state, instruction),
            aux: Poseidon2VmAuxCols {
                dst,
                lhs,
                rhs,
                internal,
            },
        }
    }
}

impl<const WIDTH: usize, F: PrimeField32> MachineChip<F> for Poseidon2Chip<WIDTH, F> {
    /// Generates final Poseidon2VmAir trace from cached rows.
    fn generate_trace(&mut self) -> RowMajorMatrix<F> {
        let row_len = self.rows.len();
        let correct_len = row_len.next_power_of_two();
        let blank_row = Poseidon2VmCols::<WIDTH, F>::blank_row(&self.air.inner).flatten();
        let diff = correct_len - row_len;
        RowMajorMatrix::new(
            self.rows
                .iter()
                .flat_map(|row| row.flatten())
                .chain(std::iter::repeat(blank_row.clone()).take(diff).flatten())
                .collect(),
            self.air.width(),
        )
    }

    fn air<SC: StarkGenericConfig>(&self) -> &dyn AnyRap<SC>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        &self.air
    }
}
