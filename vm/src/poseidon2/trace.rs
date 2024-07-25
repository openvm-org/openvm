use super::columns::*;
use crate::cpu::trace::Instruction;

use afs_primitives::is_zero::IsZeroAir;
use afs_primitives::sub_chip::LocalTraceInstructions;
use p3_air::BaseAir;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use super::{Poseidon2Chip, Poseidon2VmAir};

impl<const WIDTH: usize, F: PrimeField32> Poseidon2VmAir<WIDTH, F> {
    /// Generates a single row from inputs.
    pub fn generate_row(
        &self,
        start_timestamp: usize,
        instruction: Instruction<F>,
        dst: F,
        lhs: F,
        rhs: F,
        input_state: [F; WIDTH],
    ) -> Poseidon2VmCols<WIDTH, F> {
        // SAFETY: only allowed because WIDTH constrained to 16 above
        let internal = self.inner.generate_trace_row(input_state);
        let is_zero_row = IsZeroAir {}.generate_trace_row(instruction.d);
        Poseidon2VmCols {
            io: Poseidon2VmAir::<WIDTH, F>::make_io_cols(start_timestamp, instruction),
            aux: Poseidon2VmAuxCols {
                dst,
                lhs,
                rhs,
                d_is_zero: is_zero_row.io.is_zero,
                is_zero_inv: is_zero_row.inv,
                internal,
            },
        }
    }
}

impl<const WIDTH: usize, F: PrimeField32> Poseidon2Chip<WIDTH, F> {
    /// Generates final Poseidon2VmAir trace from cached rows.
    pub fn generate_trace(&self) -> RowMajorMatrix<F> {
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
}
