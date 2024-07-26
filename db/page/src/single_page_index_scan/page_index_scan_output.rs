use std::sync::Arc;

use afs_primitives::range_gate::RangeCheckerGateChip;
use afs_stark_backend::{air_builders::PartitionedAirBuilder, interaction::InteractionBuilder};
use itertools::Itertools;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field, PrimeField64};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_uni_stark::{StarkGenericConfig, Val};

use crate::{common::page::Page, indexed_output_page_air::IndexedOutputPageAir};

/// Wrapper around [IndexedOutputPageAir] that receives each row of the page with
/// multiplicity `is_alloc`.
pub struct PageIndexScanOutputAir {
    /// The bus index for page row receives
    pub page_bus_index: usize,
    pub inner: IndexedOutputPageAir,
}

impl PageIndexScanOutputAir {
    pub fn page_width(&self) -> usize {
        self.inner.page_width()
    }

    pub fn aux_width(&self) -> usize {
        self.inner.aux_width()
    }

    pub fn air_width(&self) -> usize {
        self.page_width() + self.aux_width()
    }
}

impl<F: Field> BaseAir<F> for PageIndexScanOutputAir {
    fn width(&self) -> usize {
        BaseAir::<F>::width(&self.inner)
    }
}

impl<AB: PartitionedAirBuilder + InteractionBuilder> Air<AB> for PageIndexScanOutputAir {
    fn eval(&self, builder: &mut AB) {
        // Making sure the page is in the proper format
        self.inner.eval(builder);

        let page = &builder.partitioned_main()[0];
        let page_local = page.row_slice(0);
        let page_blob = page_local.iter().skip(1).copied().collect_vec();
        let is_alloc = page_local[0];
        drop(page_local);

        builder.push_receive(self.page_bus_index, page_blob, is_alloc);
    }
}

/// This chip receives rows from the PageIndexScanInputChip and constrains that:
///
/// 1. All allocated rows are before unallocated rows
/// 2. The allocated rows are sorted in ascending order by index
/// 3. The allocated rows of the new page are exactly the result of the index scan (via interactions)
pub struct PageIndexScanOutputChip {
    pub air: PageIndexScanOutputAir,
    pub range_checker: Arc<RangeCheckerGateChip>,
}

impl PageIndexScanOutputChip {
    pub fn new(
        page_bus_index: usize,
        idx_len: usize,
        data_len: usize,
        idx_limb_bits: usize,
        decomp: usize,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> Self {
        Self {
            air: PageIndexScanOutputAir {
                page_bus_index,
                inner: IndexedOutputPageAir::new(
                    range_checker.bus_index(),
                    idx_len,
                    data_len,
                    idx_limb_bits,
                    decomp,
                ),
            },
            range_checker,
        }
    }

    /// Generate the trace for the page table
    pub fn gen_page_trace<SC: StarkGenericConfig>(&self, page: &Page) -> RowMajorMatrix<Val<SC>>
    where
        Val<SC>: AbstractField + PrimeField64,
    {
        page.gen_trace()
    }

    /// Generate the trace for the auxiliary columns
    pub fn gen_aux_trace<SC: StarkGenericConfig>(&self, page: &Page) -> RowMajorMatrix<Val<SC>>
    where
        Val<SC>: AbstractField + PrimeField64,
    {
        self.air
            .inner
            .gen_aux_trace::<SC>(page, self.range_checker.clone())
    }
}
