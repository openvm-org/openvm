use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};

use crate::common::page::Page;

use super::PageAir;

impl PageAir {
    /// The trace is the whole page (including the is_alloc column)
    pub fn generate_trace<SC: StarkGenericConfig>(&self, page: Page) -> RowMajorMatrix<Val<SC>>
    where
        Val<SC>: AbstractField,
    {
        RowMajorMatrix::new(
            page.rows
                .into_iter()
                .flat_map(|row| {
                    let is_alloc = Val::<SC>::from_wrapped_u32(row.is_alloc);
                    let idx = row.idx.into_iter().map(Val::<SC>::from_wrapped_u32);
                    let data = row.data.into_iter().map(Val::<SC>::from_wrapped_u32);
                    std::iter::once(is_alloc).chain(idx).chain(data)
                })
                .collect(),
            self.air_width(),
        )
    }
}
