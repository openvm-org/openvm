use p3_air::{Air, BaseAir, PairBuilder};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

use super::{columns::ProgramPreprocessedCols, ProgramAir};

impl<F: Field> BaseAir<F> for ProgramAir<F> {
    fn width(&self) -> usize {
        ProgramPreprocessedCols::<F>::get_width() + 1
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let mut rows = vec![];
        for (pc, instruction) in self.program.iter().enumerate() {
            let preprocessed_cols = ProgramPreprocessedCols {
                pc: F::from_canonical_usize(pc),
                opcode: F::from_canonical_usize(instruction.opcode as usize),
                op_a: instruction.op_a,
                op_b: instruction.op_b,
                op_c: instruction.op_c,
                as_b: instruction.d,
                as_c: instruction.e,
            };
            rows.extend(preprocessed_cols.flatten());
        }
        Some(RowMajorMatrix::new(
            rows,
            ProgramPreprocessedCols::<F>::get_width(),
        ))
    }
}

impl<AB: PairBuilder> Air<AB> for ProgramAir<AB::F> {
    fn eval(&self, _: &mut AB) {}
}
