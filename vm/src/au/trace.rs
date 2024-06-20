use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

use crate::au::columns::AUCols;
use crate::cpu::trace::ProgramExecution;

use super::AUAir;

impl AUAir {
    pub fn generate_trace<T: Field>(&self, prog_exec: &ProgramExecution<T>) -> RowMajorMatrix<T> {
        let trace = prog_exec
            .arithmetic_ops
            .iter()
            .flat_map(|op| {
                let cols = AUCols::new(op.opcode, op.operand1, op.operand2);
                cols.flatten()
            })
            .collect();

        RowMajorMatrix::new(trace, AUCols::<T>::NUM_COLS)
    }
}
