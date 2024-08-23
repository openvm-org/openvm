use std::sync::Arc;

use afs_primitives::range_gate::RangeCheckerGateChip;
use p3_field::PrimeField;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};

use super::air::FilterOutputTableAir;
use crate::common::page::Page;

impl FilterOutputTableAir {
    pub fn gen_page_trace<SC: StarkGenericConfig>(&self, page: &Page) -> RowMajorMatrix<Val<SC>>
    where
        Val<SC>: PrimeField,
    {
        page.gen_trace()
    }

    pub fn gen_aux_trace<SC: StarkGenericConfig>(
        &self,
        page: &Page,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> RowMajorMatrix<Val<SC>>
    where
        Val<SC>: PrimeField,
    {
        let mut rows: Vec<Vec<Vec<SC>>> = vec![];

        // RowMajorMatrix::new(rows.concat(), self.aux_width())
        unimplemented!()
    }
}
