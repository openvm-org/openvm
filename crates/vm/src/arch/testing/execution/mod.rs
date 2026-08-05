use std::{borrow::BorrowMut, mem::size_of};

use air::DummyExecutionInteractionCols;
use openvm_circuit_primitives::Chip;
use openvm_cpu_backend::CpuBackend;
use openvm_instructions::program::pc_to_idx;
use openvm_stark_backend::{
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    p3_matrix::dense::RowMajorMatrix,
    prover::AirProvingContext,
    StarkProtocolConfig, Val,
};

use crate::arch::{ExecutionBus, ExecutionState};

pub mod air;
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[derive(Debug)]
pub struct ExecutionTester<F: Field> {
    pub bus: ExecutionBus,
    pub records: Vec<DummyExecutionInteractionCols<F>>,
    /// The raw byte-pc states of the last execution (the records hold pc indices).
    pub last_states: Option<(ExecutionState<u32>, ExecutionState<u32>)>,
}

impl<F: PrimeField32> ExecutionTester<F> {
    pub fn new(bus: ExecutionBus) -> Self {
        Self {
            bus,
            records: vec![],
            last_states: None,
        }
    }

    /// The states carry byte pcs; the execution bus carries pc indices.
    pub fn execute(
        &mut self,
        initial_state: ExecutionState<u32>,
        final_state: ExecutionState<u32>,
    ) {
        let to_idx = |state: ExecutionState<u32>| ExecutionState {
            pc: pc_to_idx(state.pc),
            timestamp: state.timestamp,
        };
        self.records.push(DummyExecutionInteractionCols {
            count: F::NEG_ONE, // send
            initial_state: to_idx(initial_state).map(F::from_u32),
            final_state: to_idx(final_state).map(F::from_u32),
        });
        self.last_states = Some((initial_state, final_state));
    }

    /// Byte pc of the last execution's initial state. Returned as a `u32` because byte pcs
    /// span the full 32-bit range and do not fit in a field element.
    pub fn last_from_pc(&self) -> u32 {
        self.last_states.unwrap().0.pc
    }

    /// Byte pc of the last execution's final state. Returned as a `u32` because byte pcs
    /// span the full 32-bit range and do not fit in a field element.
    pub fn last_to_pc(&self) -> u32 {
        self.last_states.unwrap().1.pc
    }
}

impl<SC: StarkProtocolConfig> Chip<CpuBackend<SC>> for ExecutionTester<Val<SC>>
where
    Val<SC>: Field,
{
    fn generate_proving_ctx(&self) -> AirProvingContext<CpuBackend<SC>> {
        let height = self.records.len().next_power_of_two();
        let width = size_of::<DummyExecutionInteractionCols<u8>>();
        let mut values = Val::<SC>::zero_vec(height * width);
        // This zip only goes through records. The padding rows between records.len()..height
        // are filled with zeros - in particular count = 0 so nothing is added to bus.
        for (row, record) in values.chunks_mut(width).zip(&self.records) {
            *row.borrow_mut() = *record;
        }
        AirProvingContext::simple_no_pis(RowMajorMatrix::new(values, width))
    }
}
