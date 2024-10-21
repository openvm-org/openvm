use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use super::{columns::ProgramExecutionCols, ProgramChip};

impl<F: PrimeField64> ProgramChip<F> {
    pub fn generate_cached_trace(&self) -> RowMajorMatrix<F> {
        let mut rows = vec![];
        for (pc, instruction) in self.program.instructions().iter().enumerate() {
            let exec_cols = ProgramExecutionCols {
                pc: F::from_canonical_usize(pc),
                opcode: F::from_canonical_usize(instruction.opcode),
                a: instruction.a,
                b: instruction.b,
                c: instruction.c,
                d: instruction.d,
                e: instruction.e,
                f: instruction.f,
                g: instruction.g,
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
