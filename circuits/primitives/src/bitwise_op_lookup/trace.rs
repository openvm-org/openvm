use std::borrow::BorrowMut;

use p3_field::PrimeField;
use p3_matrix::dense::RowMajorMatrix;

use super::{
    columns::{BitwiseOperationLookupCols, NUM_BITWISE_OP_LOOKUP_COLS},
    BitwiseOperationLookupChip,
};

impl<const NUM_BITS: usize> BitwiseOperationLookupChip<NUM_BITS> {
    pub fn generate_trace<F: PrimeField>(&self) -> RowMajorMatrix<F> {
        let mut rows = vec![F::zero(); self.count_add.len() * NUM_BITWISE_OP_LOOKUP_COLS];
        for (n, row) in rows.chunks_mut(NUM_BITWISE_OP_LOOKUP_COLS).enumerate() {
            let cols: &mut BitwiseOperationLookupCols<F> = row.borrow_mut();
            cols.mult_add =
                F::from_canonical_u32(self.count_add[n].load(std::sync::atomic::Ordering::SeqCst));
            cols.mult_xor =
                F::from_canonical_u32(self.count_xor[n].load(std::sync::atomic::Ordering::SeqCst));
        }
        RowMajorMatrix::new(rows, NUM_BITWISE_OP_LOOKUP_COLS)
    }
}
