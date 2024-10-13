use std::{borrow::BorrowMut, mem::size_of, sync::Arc};

use afs_stark_backend::rap::AnyRap;
use air::{DummyExecutionInteractionCols, ExecutionDummyAir};
use p3_commit::PolynomialSpace;
use p3_field::{Field, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig};

use crate::arch::{ExecutionBus, ExecutionState, MachineChip};

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

    pub fn execute(&mut self, initial_state: ExecutionState<F>, final_state: ExecutionState<F>) {
        self.records.push(DummyExecutionInteractionCols {
            count: F::neg_one(), // send
            initial_state,
            final_state,
        })
    }

    // for use by CoreChip, needs to be modified to setup memorytester (or just merge them before writing CoreChip)
    /*fn test_execution_with_expected_changes<F: PrimeField64, E: InstructionExecutor<F>>(
        &mut self,
        executor: &mut E,
        instruction: Instruction<F>,
        expected_pc_change: usize,
        expected_timestamp_change: usize,
    ) {
        let initial_state = ExecutionState {
            pc: self.next_elem_size_usize::<F>(),
            timestamp: self.next_elem_size_usize::<F>(),
        };
        let final_state = ExecutionState {
            pc: initial_state.pc + expected_pc_change,
            timestamp: initial_state.timestamp + expected_timestamp_change,
        };
        assert_eq!(executor.execute(&instruction, initial_state), final_state);
        self.executions.push(Execution {
            initial_state,
            final_state,
            instruction: InstructionCols::from_instruction(&instruction)
                .map(|elem| elem.as_canonical_u64() as usize),
        });
    }*/
}

impl<F: Field> MachineChip<F> for ExecutionTester<F> {
    fn generate_trace(self) -> RowMajorMatrix<F> {
        let height = self.records.len().next_power_of_two();
        let width = self.trace_width();
        let mut values = vec![F::zero(); height * width];
        // This zip only goes through records. The padding rows between records.len()..height
        // are filled with zeros - in particular count = 0 so nothing is added to bus.
        for (row, record) in values.chunks_mut(width).zip(&self.records) {
            *row.borrow_mut() = *record;
        }
        RowMajorMatrix::new(values, self.trace_width())
    }

    fn air<SC: StarkGenericConfig>(&self) -> Arc<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        Arc::new(ExecutionDummyAir::new(self.bus))
    }

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
