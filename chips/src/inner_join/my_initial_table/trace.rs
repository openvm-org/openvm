use afs_test_utils::utils::to_field_vec;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

use super::MyInitialTableAir;

impl MyInitialTableAir {
    pub fn gen_aux_trace<F: Field>(&self, out_mult: &[u32]) -> RowMajorMatrix<F> {
        RowMajorMatrix::new_col(to_field_vec(out_mult.to_vec()))
    }
}
