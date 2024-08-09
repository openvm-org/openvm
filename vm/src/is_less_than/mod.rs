use std::sync::Arc;

use afs_primitives::{
    is_less_than::{columns::IsLessThanCols, IsLessThanAir},
    range_gate::RangeCheckerGateChip,
    sub_chip::LocalTraceInstructions,
};
use columns::IsLessThanVmCols;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

pub mod air;
pub mod bridge;
pub mod columns;
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
    pub rows: Vec<IsLessThanVmCols<F>>,
}

impl<F: PrimeField32> IsLessThanChip<F> {
    pub fn new(
        bus_index: usize,
        max_bits: usize,
        decomp: usize,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> Self {
        Self {
            air: IsLessThanVmAir {
                bus_index,
                inner: IsLessThanAir::new(range_checker.air.bus_index, max_bits, decomp),
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
        self.rows.push(IsLessThanVmCols {
            is_enabled: F::one(),
            internal: row,
        });

        result
    }

    pub fn generate_trace(&self) -> RowMajorMatrix<F> {
        let width = IsLessThanCols::<F>::width(&self.air.inner) + 1;
        let mut traces: Vec<F> = self.rows.iter().flat_map(|row| row.flatten()).collect();
        let current_height = self.rows.len();
        let correct_height = current_height.next_power_of_two();
        traces.resize(correct_height * width, F::zero());
        RowMajorMatrix::new(traces, width)
    }
}
