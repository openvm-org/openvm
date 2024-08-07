use std::sync::Arc;

use afs_primitives::{
    is_less_than::{columns::IsLessThanCols, IsLessThanAir},
    range_gate::RangeCheckerGateChip,
    sub_chip::LocalTraceInstructions,
};
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

pub mod air;
pub mod bridge;
#[cfg(test)]
pub mod tests;

#[derive(Clone, Copy)]
pub struct IsLessThanVmAir {
    pub bus_index: usize,
    pub inner: IsLessThanAir,
}

pub struct IsLessThanChip<F: PrimeField32> {
    pub air: IsLessThanVmAir,
    pub range_checker: Arc<RangeCheckerGateChip>,
    pub rows: Vec<IsLessThanCols<F>>,
}

impl<F: PrimeField32> IsLessThanChip<F> {
    pub fn new(
        bus_index: usize,               // CPU <> IsLessThanChip
        range_checker_bus_index: usize, // IsLessThanChip <> RangeChecker
        max_bits: usize,
        decomp: usize,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> Self {
        Self {
            air: IsLessThanVmAir {
                bus_index,
                inner: IsLessThanAir::new(range_checker_bus_index, max_bits, decomp),
            },
            range_checker,
            rows: Vec::new(),
        }
    }

    // Returns the result, and save the operations for trace generation.
    pub fn compare(&mut self, operands: (F, F)) -> F {
        let x = operands.0.as_canonical_u32();
        let y = operands.1.as_canonical_u32();
        let row = LocalTraceInstructions::<F>::generate_trace_row(
            &self.air.inner,
            (x, y, self.range_checker.clone()),
        );
        let result = row.io.less_than;
        self.rows.push(row);

        result
    }

    pub fn generate_trace(&self) -> RowMajorMatrix<F> {
        let width = IsLessThanCols::<F>::width(&self.air.inner);
        let mut traces: Vec<F> = self.rows.iter().flat_map(|row| row.flatten()).collect();
        // Pad empty rows so the height is a power of 2.
        let empty_row: Vec<F> = vec![F::zero(); width];
        let current_height = self.rows.len();
        let correct_height = current_height.next_power_of_two();
        traces.extend(
            empty_row
                .iter()
                .cloned()
                .cycle()
                .take((correct_height - current_height) * width),
        );
        RowMajorMatrix::new(traces, width)
    }
}
