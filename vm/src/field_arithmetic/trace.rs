use std::{array, vec::IntoIter};

use afs_stark_backend::rap::AnyRap;
use p3_commit::PolynomialSpace;
use p3_field::{Field, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig};

use super::{
    columns::{FieldArithmeticAuxCols, FieldArithmeticCols, FieldArithmeticIoCols},
    FieldArithmeticChip, FieldArithmeticOperation, Operand,
};
use crate::{
    arch::{chips::MachineChip, columns::ExecutionState, instructions::Opcode},
    memory::{
        manager::trace_builder::MemoryTraceBuilder,
        offline_checker::columns::MemoryOfflineCheckerAuxCols, OpType,
    },
};

fn generate_row<F: Field>(
    operation: FieldArithmeticOperation<F>,
    is_valid: bool,
    oc_aux_iter: &mut IntoIter<MemoryOfflineCheckerAuxCols<1, F>>,
) -> FieldArithmeticCols<F> {
    let FieldArithmeticOperation {
        opcode,
        from_state,
        operand1,
        operand2,
        result,
    } = operation;

    let is_add = F::from_bool(opcode == Opcode::FADD);
    let is_sub = F::from_bool(opcode == Opcode::FSUB);
    let is_div = F::from_bool(opcode == Opcode::FDIV);
    let is_mul = F::from_bool(opcode == Opcode::FMUL);
    let divisor_inv = if opcode == Opcode::FDIV {
        operand2.value.inverse()
    } else {
        F::zero()
    };

    FieldArithmeticCols {
        io: FieldArithmeticIoCols {
            opcode: F::from_canonical_u32(opcode as u32),
            from_state: from_state.map(F::from_canonical_usize),
            operand1,
            operand2,
            result,
        },
        aux: FieldArithmeticAuxCols {
            is_valid: F::from_bool(is_valid),
            is_add,
            is_sub,
            is_mul,
            is_div,
            divisor_inv,
            mem_oc_aux_cols: array::from_fn(|_| oc_aux_iter.next().unwrap()),
        },
    }
}

impl<F: PrimeField32> FieldArithmeticChip<F> {
    fn make_blank_row(&self) -> FieldArithmeticCols<F> {
        let mut trace_builder = MemoryTraceBuilder::new(self.memory_manager.clone());

        let timestamp = self
            .memory_manager
            .borrow_mut()
            .timestamp()
            .as_canonical_u32() as usize;

        trace_builder.disabled_op(F::one(), OpType::Read);
        trace_builder.disabled_op(F::one(), OpType::Read);
        trace_builder.disabled_op(F::one(), OpType::Write);
        let mut mem_oc_aux_iter = trace_builder.take_accesses_buffer().into_iter();

        let blank_cell = Operand::new(F::one(), F::zero(), F::zero());

        generate_row(
            FieldArithmeticOperation {
                opcode: Opcode::FADD,
                from_state: ExecutionState { pc: 0, timestamp },
                operand1: blank_cell,
                operand2: blank_cell,
                result: blank_cell,
            },
            false,
            &mut mem_oc_aux_iter,
        )
    }
}

impl<F: PrimeField32> MachineChip<F> for FieldArithmeticChip<F> {
    /// Generates trace for field arithmetic chip.
    fn generate_trace(&mut self) -> RowMajorMatrix<F> {
        let accesses = self.memory.take_accesses_buffer();
        let mut accesses_iter = accesses.into_iter();

        let mut trace: Vec<F> = self
            .operations
            .iter()
            .flat_map(|&op| generate_row(op, true, &mut accesses_iter).flatten())
            .collect();

        let curr_height = self.operations.len();
        let correct_height = curr_height.next_power_of_two();
        // WARNING: do not clone below because timestamps are different per row
        let dummy_rows_flattened =
            (0..correct_height - curr_height).flat_map(|_| self.make_blank_row().flatten());
        trace.extend(dummy_rows_flattened);

        RowMajorMatrix::new(trace, self.trace_width())
    }

    fn air<SC: StarkGenericConfig>(&self) -> Box<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        Box::new(self.air)
    }

    fn current_trace_height(&self) -> usize {
        self.operations.len()
    }

    fn trace_width(&self) -> usize {
        FieldArithmeticCols::<F>::get_width(&self.air)
    }
}
