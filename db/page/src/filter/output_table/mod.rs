use std::sync::Arc;

use afs_primitives::range_gate::RangeCheckerGateChip;
use p3_field::{AbstractField, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};

use self::air::FilterOutputTableAir;
use crate::{common::page::Page, indexed_output_page_air::IndexedOutputPageAir};

pub mod air;
pub mod bridge;

pub struct FilterOutputTableChip {
    pub air: FilterOutputTableAir,
    pub range_checker: Arc<RangeCheckerGateChip>,
}

impl FilterOutputTableChip {
    pub fn new(
        page_bus_index: usize,
        idx_len: usize,
        data_len: usize,
        idx_limb_bits: usize,
        idx_decomp: usize,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> Self {
        Self {
            air: FilterOutputTableAir {
                page_bus_index,
                final_air: IndexedOutputPageAir::new(
                    range_checker.bus_index(),
                    idx_len,
                    data_len,
                    idx_limb_bits,
                    idx_decomp,
                ),
            },
            range_checker,
        }
    }

    pub fn gen_page_trace<SC: StarkGenericConfig>(&self, page: &Page) -> RowMajorMatrix<Val<SC>>
    where
        Val<SC>: AbstractField + PrimeField64,
    {
        page.gen_trace()
    }

    pub fn gen_aux_trace<SC: StarkGenericConfig>(&self, page: &Page) -> RowMajorMatrix<Val<SC>>
    where
        Val<SC>: AbstractField + PrimeField64,
    {
        self.air
            .final_air
            .gen_aux_trace::<SC>(page, self.range_checker.clone())
    }
}
