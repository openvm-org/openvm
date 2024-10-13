use std::sync::Arc;

use afs_stark_backend::{
    config::{StarkGenericConfig, Val},
    rap::{get_air_name, AnyRap},
    Chip,
};
use p3_air::BaseAir;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use super::{columns::*, Poseidon2Chip};
use crate::arch::MachineChip;

impl<F: PrimeField32> MachineChip<F> for Poseidon2Chip<F> {
    /// Generates final Poseidon2VmAir trace from cached rows.
    fn generate_trace(self) -> RowMajorMatrix<F> {
        let Self {
            air,
            memory_chip,
            records,
            offset: _,
        } = self;

        let row_len = records.len();
        let correct_len = row_len.next_power_of_two();
        let diff = correct_len - row_len;

        let aux_cols_factory = memory_chip.borrow().aux_cols_factory();
        let mut flat_rows: Vec<_> = records
            .into_iter()
            .flat_map(|record| Self::record_to_cols(&aux_cols_factory, record).flatten())
            .collect();
        for _ in 0..diff {
            flat_rows.extend(Poseidon2VmCols::<F>::blank_row(&air).flatten());
        }

        RowMajorMatrix::new(flat_rows, air.width())
    }

    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        self.records.len()
    }

    fn trace_width(&self) -> usize {
        self.air.width()
    }
}

impl<SC: StarkGenericConfig> Chip<SC> for Poseidon2Chip<Val<SC>>
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        Arc::new(self.air.clone())
    }
}
