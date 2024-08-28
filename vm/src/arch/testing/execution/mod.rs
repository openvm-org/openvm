use std::{borrow::BorrowMut, mem::size_of};

use afs_stark_backend::rap::AnyRap;
use air::{DummyExecutionInteractionCols, ExecutionDummyAir};
use p3_commit::PolynomialSpace;
use p3_field::{Field, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig};
use rand::{rngs::StdRng, RngCore};

use super::MemoryTester;
use crate::{
    arch::{
        bus::ExecutionBus,
        chips::{InstructionExecutor, MachineChip},
        columns::{ExecutionState, InstructionCols},
    },
    cpu::trace::Instruction,
};

pub mod air;

#[derive(Clone, Debug)]
pub struct ExecutionTester<F: Field> {
    pub bus: ExecutionBus,
    pub rng: StdRng,
    pub records: Vec<DummyExecutionInteractionCols<F>>,
}

impl<F: PrimeField32> ExecutionTester<F> {
    pub fn new(bus: ExecutionBus, rng: StdRng) -> Self {
        Self {
            bus,
            rng,
            records: vec![],
        }
    }

    pub fn execute<E: InstructionExecutor<F>>(
        &mut self,
        memory_tester: &mut MemoryTester<F>, // should merge MemoryTester and ExecutionTester into one struct (MachineChipTestBuilder?)
        executor: &mut E,
        instruction: Instruction<F>,
    ) {
        let initial_state = ExecutionState {
            pc: self.next_elem_size_usize(),
            timestamp: memory_tester.chip.borrow().timestamp(),
        };
        tracing::debug!(?initial_state.timestamp);

        let final_state = executor.execute(
            &instruction,
            initial_state.map(|x| x.as_canonical_u32() as usize),
        );
        self.records.push(DummyExecutionInteractionCols {
            count: F::neg_one(), // send
            initial_state,
            final_state: final_state.map(F::from_canonical_usize),
            instruction: InstructionCols::from_instruction(&instruction),
        })
    }

    fn next_elem_size_usize(&mut self) -> F {
        F::from_canonical_u32(self.rng.next_u32() % (1 << (F::bits() - 2)))
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
        for (row, record) in values.chunks_mut(width).into_iter().zip(&self.records) {
            *row.borrow_mut() = *record;
        }
        RowMajorMatrix::new(values, self.trace_width())
    }

    fn air<SC: StarkGenericConfig>(&self) -> Box<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        Box::new(ExecutionDummyAir::new(self.bus))
    }

    fn current_trace_height(&self) -> usize {
        self.records.len()
    }

    fn trace_width(&self) -> usize {
        size_of::<DummyExecutionInteractionCols<u8>>()
    }
}
