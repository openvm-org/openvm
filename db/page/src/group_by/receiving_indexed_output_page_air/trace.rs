use std::sync::Arc;

use afs_primitives::var_range::VariableRangeCheckerChip;
use p3_field::PrimeField;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};

use super::ReceivingIndexedOutputPageAir;
use crate::common::page::Page;

impl ReceivingIndexedOutputPageAir {
    /// Naked trace of only the page, including the is_alloc column
    pub fn gen_page_trace<SC: StarkGenericConfig>(&self, page: &Page) -> RowMajorMatrix<Val<SC>>
    where
        Val<SC>: PrimeField,
    {
        self.final_air.gen_page_trace::<SC>(page)
    }

    /// As a minimal wrapper of [FinalPageAir], generates the auxiliary trace required to ensure proper formating
    /// of the page using FinalPageAir. Includes allocated rows.
    pub fn gen_aux_trace<SC: StarkGenericConfig>(
        &self,
        page: &Page,
        range_checker: Arc<VariableRangeCheckerChip>,
    ) -> RowMajorMatrix<Val<SC>>
    where
        Val<SC>: PrimeField,
    {
        self.final_air.gen_aux_trace::<SC>(page, range_checker)
    }
}
