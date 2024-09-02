use afs_primitives::utils::to_field_vec;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

use super::InitialTableAir;

impl InitialTableAir {
    pub fn gen_aux_trace<F: Field>(&self, out_mult: &[u32]) -> RowMajorMatrix<F> {
        RowMajorMatrix::new_col(to_field_vec(out_mult))
    }
}
