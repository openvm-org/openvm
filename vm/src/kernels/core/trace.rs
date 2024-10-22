use std::{iter, sync::Arc};

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
    /// Pad with NOP rows.
    pub fn pad_rows(&mut self) {
        let curr_height = self.current_trace_height();
        let padded_height = curr_height.next_power_of_two();
        let blank_row = self.make_blank_row();
        self.flatten_rows.extend(
            iter::repeat(blank_row.flatten())
                .take(padded_height - curr_height)
                .flatten(),
        );
    }

    /// This must be called for each blank row and results should never be cloned; see [CoreCols::nop_row].
    fn make_blank_row(&self) -> CoreCols<F> {
        CoreCols::nop_row(0)
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

        let air = self.air();
        let trace = RowMajorMatrix::new(self.flatten_rows, CoreCols::<Val<SC>>::get_width());
        AirProofInput::simple_no_pis(air, trace)
    }
}

impl<F: PrimeField32> ChipUsageGetter for CoreChip<F> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }
    fn current_trace_height(&self) -> usize {
        self.flatten_rows.len() / self.trace_width()
    }
    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}
