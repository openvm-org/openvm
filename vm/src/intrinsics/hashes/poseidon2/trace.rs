use std::sync::Arc;

use afs_primitives::utils::next_power_of_two_or_zero;
use ax_stark_backend::{
    config::{StarkGenericConfig, Val},
    prover::types::AirProofInput,
    rap::{get_air_name, AnyRap},
    Chip, ChipUsageGetter,
};
use p3_air::BaseAir;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use super::{columns::*, Poseidon2Chip};

impl<SC: StarkGenericConfig> Chip<SC> for Poseidon2Chip<Val<SC>>
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        Arc::new(self.air.clone())
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        let Self {
            air,
            memory_controller,
            records,
            offset: _,
        } = self;

        let row_len = records.len();
        let correct_len = next_power_of_two_or_zero(row_len);
        let diff = correct_len - row_len;

        let aux_cols_factory = memory_controller.borrow().aux_cols_factory();
        let mut flat_rows: Vec<_> = records
            .into_iter()
            .flat_map(|record| Self::record_to_cols(&aux_cols_factory, record).flatten())
            .collect();
        for _ in 0..diff {
            flat_rows.extend(Poseidon2VmCols::<Val<SC>>::blank_row(&air).flatten());
        }

        AirProofInput::simple_no_pis(
            Arc::new(air.clone()),
            RowMajorMatrix::new(flat_rows, air.width()),
        )
    }
}

impl<F: PrimeField32> ChipUsageGetter for Poseidon2Chip<F> {
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
