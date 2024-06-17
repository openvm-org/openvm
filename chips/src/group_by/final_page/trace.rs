use crate::common::page::Page;
use std::sync::Arc;

use p3_field::PrimeField;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};

use super::MyFinalPageAir;
use crate::range_gate::RangeCheckerGateChip;

impl MyFinalPageAir {
    /// The trace is the whole page (including the is_alloc column)
    pub fn gen_page_trace<SC: StarkGenericConfig>(&self, page: &Page) -> RowMajorMatrix<Val<SC>>
    where
        Val<SC>: PrimeField,
    {
        self.final_air.gen_page_trace::<SC>(page)
    }

    /// This generates the auxiliary trace required to ensure proper formating
    /// of the page using FinalPageAir. Moreover, it generates the rcv_mult column, which is on
    /// only when the index is in internal_indices and is allocated in the page
    /// Here, internal_indices is a set of indices that appear in the operations
    pub fn gen_aux_trace<SC: StarkGenericConfig>(
        &self,
        page: &Page,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> RowMajorMatrix<Val<SC>>
    where
        Val<SC>: PrimeField,
    {
        self.final_air.gen_aux_trace::<SC>(page, range_checker)
    }
}
