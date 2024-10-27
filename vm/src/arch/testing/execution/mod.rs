use std::{borrow::BorrowMut, mem::size_of, sync::Arc};

use air::{DummyExecutionInteractionCols, ExecutionDummyAir};
use ax_stark_backend::{
    config::{StarkGenericConfig, Val},
    prover::types::AirProofInput,
    rap::AnyRap,
    Chip, ChipUsageGetter,
};
use p3_field::{AbstractField, Field, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;

use crate::arch::{ExecutionBus, ExecutionState};

pub mod air;

#[derive(Clone, Debug)]
pub struct ExecutionTester<F: Field> {
    pub bus: ExecutionBus,
    pub records: Vec<DummyExecutionInteractionCols<F>>,
}

impl<F: PrimeField32> ExecutionTester<F> {
    pub fn new(bus: ExecutionBus) -> Self {
        Self {
            bus,
            records: vec![],
        }
    }

    pub fn execute(
        &mut self,
        initial_state: ExecutionState<u32>,
        final_state: ExecutionState<u32>,
    ) {
        self.records.push(DummyExecutionInteractionCols {
            count: F::neg_one(), // send
            initial_state: initial_state.map(F::from_canonical_u32),
            final_state: final_state.map(F::from_canonical_u32),
        })
    }

    pub fn last_from_pc(&self) -> F {
        self.records.last().unwrap().initial_state.pc
    }

    pub fn last_to_pc(&self) -> F {
        self.records.last().unwrap().final_state.pc
    }
}

impl<SC: StarkGenericConfig> Chip<SC> for ExecutionTester<Val<SC>>
where
    Val<SC>: Field,
{
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        Arc::new(ExecutionDummyAir::new(self.bus))
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        let air = self.air();
        let height = self.records.len().next_power_of_two();
        let width = self.trace_width();
        let mut values = vec![Val::<SC>::zero(); height * width];
        // This zip only goes through records. The padding rows between records.len()..height
        // are filled with zeros - in particular count = 0 so nothing is added to bus.
        for (row, record) in values.chunks_mut(width).zip(self.records) {
            *row.borrow_mut() = record;
        }
        AirProofInput::simple_no_pis(air, RowMajorMatrix::new(values, width))
    }
}
impl<F: Field> ChipUsageGetter for ExecutionTester<F> {
    fn air_name(&self) -> String {
        "ExecutionDummyAir".to_string()
    }
    fn current_trace_height(&self) -> usize {
        self.records.len()
    }

    fn trace_width(&self) -> usize {
        size_of::<DummyExecutionInteractionCols<u8>>()
    }
}
