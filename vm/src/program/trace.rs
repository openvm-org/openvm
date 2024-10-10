use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use super::{columns::ProgramExecutionCols, ProgramChip};

impl<F: PrimeField64> ProgramChip<F> {
    pub fn generate_cached_trace(&self) -> RowMajorMatrix<F> {
        let mut rows = vec![];
        for (pc, instruction) in self.air.program.instructions.iter().enumerate() {
            let exec_cols = ProgramExecutionCols {
                pc: F::from_canonical_usize(pc),
                opcode: F::from_canonical_usize(instruction.opcode),
                op_a: instruction.op_a,
                op_b: instruction.op_b,
                op_c: instruction.op_c,
                as_b: instruction.d,
                as_c: instruction.e,
                op_f: instruction.op_f,
                op_g: instruction.op_g,
            };
            rows.extend(exec_cols.flatten());
        }

        RowMajorMatrix::new(rows, ProgramExecutionCols::<F>::width())
    }

    pub fn generate_trace(self) -> RowMajorMatrix<F> {
        RowMajorMatrix::new_col(
            self.execution_frequencies
                .iter()
                .map(|x| F::from_canonical_usize(*x))
                .collect::<Vec<F>>(),
        )
    }
}
