use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

use crate::cpu::trace::ProgramExecution;
use crate::fp4_extension::add_sub::columns::FieldExtensionAddSubCols;

use super::FieldExtensionAddSubAir;

impl FieldExtensionAddSubAir {
    /// Generates trace for field arithmetic chip.
    pub fn generate_trace<T: Field>(&self, prog_exec: &ProgramExecution<T>) -> RowMajorMatrix<T> {
        let trace: Vec<T> = prog_exec
            .field_extension_ops
            .iter()
            .flat_map(|op| {
                let cols = FieldExtensionAddSubCols::from_slice(&op.to_vec());
                cols.flatten()
            })
            .collect();

        RowMajorMatrix::new(trace, FieldExtensionAddSubCols::<T>::NUM_COLS)
    }
}
