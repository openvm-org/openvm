use std::sync::Arc;

use afs_stark_backend::{
    config::{StarkGenericConfig, Val},
    prover::types::AirProofInput,
    rap::{get_air_name, AnyRap},
    Chip, ChipUsageGetter,
};
use p3_air::BaseAir;
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;

use super::{columns::CoreCols, CoreChip};

impl<F: PrimeField32> CoreChip<F> {
    /// Pad with NOP rows.
    pub fn pad_rows(&mut self) {
        let curr_height = self.rows.len();
        let correct_height = self.rows.len().next_power_of_two();
        for _ in 0..correct_height - curr_height {
            self.rows.push(self.make_blank_row().flatten());
        }
    }

    /// This must be called for each blank row and results should never be cloned; see [CoreCols::nop_row].
    fn make_blank_row(&self) -> CoreCols<F> {
        let pc = F::from_canonical_u32(self.state.pc);
        let timestamp = F::from_canonical_u32(self.state.timestamp);
        CoreCols::nop_row(self, pc, timestamp)
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

        let trace = RowMajorMatrix::new(
            self.rows.concat(),
            CoreCols::<Val<SC>>::get_width(&self.air),
        );
        let public_values = {
            let first_row_pc = self.start_state.pc;
            let last_row_pc = self.state.pc;
            let mut result = vec![
                Val::<SC>::from_canonical_u32(first_row_pc),
                Val::<SC>::from_canonical_u32(last_row_pc),
            ];
            result.extend(
                self.public_values
                    .iter()
                    .map(|pv| pv.unwrap_or(Val::<SC>::zero())),
            );
            result
        };
        AirProofInput::simple(self.air(), trace, public_values)
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
