use std::{collections::HashSet, sync::Arc};

use p3_field::{AbstractField, PrimeField};
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};

use super::IndexedPageWriteAir;
use crate::{common::page::Page, range_gate::RangeCheckerGateChip};

impl IndexedPageWriteAir {
    /// This generates the auxiliary trace required to ensure proper formating
    /// of the page using FinalPageAir. Moreover, it generates the rcv_mult column, which is on
    /// only when the index is in internal_indices and is allocated in the page
    /// Here, internal_indices is a set of indices that appear in the operations
    pub fn gen_aux_trace<SC: StarkGenericConfig>(
        &self,
        page: &Page,
        range_checker: Arc<RangeCheckerGateChip>,
        internal_indices: HashSet<Vec<u32>>,
    ) -> RowMajorMatrix<Val<SC>>
    where
        Val<SC>: PrimeField,
    {
        let mut final_page_aux_trace = self.final_air.gen_aux_trace::<SC>(page, range_checker);

        let mut aux_trace_flat: Vec<Val<SC>> = vec![];
        for (r, page_row) in page.rows.iter().enumerate() {
            let fp_aux_row = final_page_aux_trace.row_mut(r);
            aux_trace_flat.extend_from_slice(fp_aux_row);

            let cur_idx = page_row.idx.clone();
            aux_trace_flat.push(Val::<SC>::from_canonical_u8(
                if internal_indices.contains(&cur_idx) && page_row.is_alloc == 1 {
                    3
                } else {
                    page_row.is_alloc as u8
                },
            ));
        }

        RowMajorMatrix::new(aux_trace_flat, self.aux_width())
    }
}
