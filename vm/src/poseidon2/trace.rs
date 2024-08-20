use p3_air::BaseAir;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use super::{columns::*, Poseidon2Chip};
use crate::memory::{manager::trace_builder::MemoryTraceBuilder, OpType};

impl<const WIDTH: usize, const NUM_WORDS: usize, const WORD_SIZE: usize, F: PrimeField32>
    Poseidon2Chip<WIDTH, NUM_WORDS, WORD_SIZE, F>
{
    /// Generates final Poseidon2VmAir trace from cached rows.
    pub fn generate_trace(&self) -> RowMajorMatrix<F> {
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

    pub fn blank_row(&self) -> Poseidon2VmCols<WIDTH, WORD_SIZE, F> {
        let timestamp = self.memory_manager.borrow().get_clk();
        let mut blank =
            Poseidon2VmCols::<WIDTH, WORD_SIZE, F>::blank_row(&self.air.inner, timestamp);
        let mut mem_trace_builder = MemoryTraceBuilder::new(
            self.memory_manager.clone(),
            self.range_checker.clone(),
            self.air.mem_oc,
        );
        for _ in 0..3 {
            mem_trace_builder.disabled_op(blank.io.d, OpType::Read);
            mem_trace_builder.increment_clk();
        }
        for _ in 0..WIDTH {
            mem_trace_builder.disabled_op(blank.io.e, OpType::Read);
            mem_trace_builder.increment_clk();
        }
        for _ in 0..WIDTH {
            mem_trace_builder.disabled_op(blank.io.e, OpType::Write);
            mem_trace_builder.increment_clk();
        }
        blank
            .aux
            .mem_oc_aux_cols
            .extend(mem_trace_builder.take_accesses_buffer());

        blank
    }
}
