use super::PageReadChip;
use p3_field::AbstractField;
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix};
use p3_uni_stark::{StarkGenericConfig, Val};

impl PageReadChip {
    pub fn get_page_trace<SC: StarkGenericConfig>(&self) -> DenseMatrix<Val<SC>>
    where
        Val<SC>: AbstractField,
    {
        assert!(!self.page_data.is_empty());

        let val_len = self.page_data[0].len();
        RowMajorMatrix::new(
            self.page_data
                .clone()
                .into_iter()
                .flat_map(|row| row.into_iter().map(Val::<SC>::from_wrapped_u32))
                .collect(),
            val_len,
        )
    }
}
