use std::sync::Arc;

use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};

use crate::{prover::types::AirProofInput, rap::AnyRap};

/// See [Chip]. This trait contains the methods of a chip that do not
/// require knowing the STARK config. Only the generic `F` is required.
pub trait BaseChip<F>: Sized {
    fn generate_trace(self) -> RowMajorMatrix<F>;
    fn generate_public_values(&mut self) -> Vec<F> {
        vec![]
    }
    fn current_trace_height(&self) -> usize;
    /// Width of underlying AIR in base field columns
    fn trace_width(&self) -> usize;

    /// For metrics collection
    fn current_trace_cells(&self) -> usize {
        self.trace_width() * self.current_trace_height()
    }

    /// Name of the underlying AIR, for debugging purposes
    fn air_name(&self) -> String;
}

/// A chip is a stateful struct that stores the state necessary to
/// generate the trace of an AIR. This trait is for proving purposes
/// and has a generic [StarkGenericConfig] since it needs to know the STARK config.
pub trait Chip<SC: StarkGenericConfig>: BaseChip<Val<SC>> {
    fn air(&self) -> Arc<dyn AnyRap<SC>>;
    /// Generate all necessary input for proving a single AIR.
    fn generate_air_proof_input(&self) -> AirProofInput<SC> {
        // TEMPORARY[jpw]: make it easier to implement this trait in transition
        todo!();
    }
    fn generate_air_proof_input_with_id(&self, air_id: usize) -> (usize, AirProofInput<SC>) {
        (air_id, self.generate_air_proof_input())
    }
}
