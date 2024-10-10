use std::{borrow::BorrowMut, mem::size_of};

use afs_stark_backend::rap::AnyRap;
use air::ProgramDummyAir;
use p3_commit::PolynomialSpace;
use p3_field::{Field, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig};

use crate::{
    arch::{chips::MachineChip, ExecutionState},
    program::{bridge::ProgramBus, columns::ProgramExecutionCols, Instruction},
};

mod air;

#[derive(Clone, Debug)]
pub struct ProgramTester<F: Field> {
    pub bus: ProgramBus,
    pub records: Vec<ProgramExecutionCols<F>>,
}

impl<F: PrimeField32> ProgramTester<F> {
    pub fn new(bus: ProgramBus) -> Self {
        Self {
            bus,
            records: vec![],
        }
    }

    pub fn execute(&mut self, instruction: Instruction<F>, initial_state: &ExecutionState<F>) {
        self.records.push(ProgramExecutionCols {
            pc: initial_state.pc,
            opcode: F::from_canonical_u8(instruction.opcode as u8),
            op_a: instruction.op_a,
            op_b: instruction.op_b,
            op_c: instruction.op_c,
            as_b: instruction.d,
            as_c: instruction.e,
            op_f: instruction.op_f,
            op_g: instruction.op_g,
        });
    }
}

impl<F: Field> ProgramTester<F> {
    fn width() -> usize {
        size_of::<ProgramExecutionCols<u8>>() + 1
    }
}

impl<F: Field> MachineChip<F> for ProgramTester<F> {
    fn generate_trace(self) -> RowMajorMatrix<F> {
        let height = self.records.len().next_power_of_two();
        let width = self.trace_width();
        let mut values = vec![F::zero(); height * width];
        // This zip only goes through records. The padding rows between records.len()..height
        // are filled with zeros - in particular count = 0 so nothing is added to bus.
        for (row, record) in values.chunks_mut(width).zip(&self.records) {
            *(row[..width - 1]).borrow_mut() = *record;
            row[width - 1] = F::one();
        }
        RowMajorMatrix::new(values, width)
    }

    fn air<SC: StarkGenericConfig>(&self) -> Box<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        Box::new(ProgramDummyAir::new(self.bus))
    }

    fn air_name(&self) -> String {
        "ProgramDummyAir".to_string()
    }

    fn current_trace_height(&self) -> usize {
        self.records.len()
    }

    fn trace_width(&self) -> usize {
        Self::width()
    }
}
