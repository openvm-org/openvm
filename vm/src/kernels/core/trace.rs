use std::sync::Arc;

use afs_stark_backend::{
    config::{StarkGenericConfig, Val},
    prover::types::AirProofInput,
    rap::{get_air_name, AnyRap},
    Chip, ChipUsageGetter,
};
use p3_air::BaseAir;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use super::{columns::CoreCols, CoreChip};

impl<F: PrimeField32> CoreChip<F> {
    /// Pad with blank rows.
    pub fn pad_rows(&mut self) {
        let curr_height = self.rows.len();
        let mut correct_height = self.rows.len().next_power_of_two();
        if correct_height == 1 {
            correct_height = 2;
        }
        for _ in 0..correct_height - curr_height {
            self.rows.push(CoreCols::blank_row().flatten());
        }
    }
}

impl<SC: StarkGenericConfig> Chip<SC> for CoreChip<Val<SC>>
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        Arc::new(self.air.clone())
    }

    fn generate_air_proof_input(mut self) -> AirProofInput<SC> {
        self.pad_rows();

        let trace = RowMajorMatrix::new(self.rows.concat(), CoreCols::<Val<SC>>::get_width());
        AirProofInput::simple_no_pis(self.air(), trace)
    }
}

impl<F: PrimeField32> ChipUsageGetter for CoreChip<F> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }
    fn current_trace_height(&self) -> usize {
        self.rows.len()
    }
    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}
