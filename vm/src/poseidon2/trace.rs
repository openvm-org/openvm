use afs_stark_backend::rap::AnyRap;
use p3_air::BaseAir;
use p3_commit::PolynomialSpace;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig};

use super::{columns::*, Poseidon2Chip};
use crate::{arch::chips::MachineChip, memory::manager::trace_builder::MemoryTraceBuilder};

impl<const WIDTH: usize, F: PrimeField32> MachineChip<F> for Poseidon2Chip<WIDTH, F> {
    /// Generates final Poseidon2VmAir trace from cached rows.
    fn generate_trace(&mut self) -> RowMajorMatrix<F> {
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
    pub fn blank_row(&self) -> Poseidon2VmCols<WIDTH, F> {
        let timestamp = self.memory_chip.borrow().timestamp();
        let mut blank = Poseidon2VmCols::<WIDTH, F>::blank_row(&self.air.inner, timestamp);
        let mut mem_trace_builder = MemoryTraceBuilder::new(self.memory_chip.clone());
        for _ in 0..3 {
            mem_trace_builder.disabled_read(blank.io.d);
            mem_trace_builder.increment_clk();
        }
        for _ in 0..WIDTH {
            mem_trace_builder.disabled_read(blank.io.e);
            mem_trace_builder.increment_clk();
        }
        for _ in 0..WIDTH {
            mem_trace_builder.disabled_write(blank.io.e);
            mem_trace_builder.increment_clk();
        }
        blank
            .aux
            .mem_oc_aux_cols
            .extend(mem_trace_builder.take_accesses_buffer());

        blank
    }
}
