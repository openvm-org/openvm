use std::sync::Arc;

use p3_field::{AbstractField, PrimeField};
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};

use super::{columns::FinalPageAuxCols, FinalPageAir};
use crate::{
    is_less_than_tuple::{columns::IsLessThanTupleCols, IsLessThanTupleAir},
    range_gate::RangeCheckerGateChip,
    sub_chip::LocalTraceInstructions,
};

impl FinalPageAir {
    /// The trace is the whole page (including the is_alloc column)
    pub fn gen_page_trace<SC: StarkGenericConfig>(
        &self,
        page: Vec<Vec<u32>>,
    ) -> RowMajorMatrix<Val<SC>>
    where
        Val<SC>: AbstractField,
    {
        RowMajorMatrix::new(
            page.into_iter()
                .flat_map(|row| {
                    row.into_iter()
                        .map(Val::<SC>::from_wrapped_u32)
                        .collect::<Vec<Val<SC>>>()
                })
                .collect(),
            self.page_width(),
        )
    }

    /// This generates the auxiliary trace required to ensure proper formating
    /// of the page
    pub fn gen_aux_trace<SC: StarkGenericConfig>(
        &self,
        page: Vec<Vec<u32>>,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> RowMajorMatrix<Val<SC>>
    where
        Val<SC>: PrimeField,
    {
        let lt_chip = IsLessThanTupleAir::new(
            self.range_bus_index,
            1 << self.idx_limb_bits,
            vec![self.idx_limb_bits; self.idx_len],
            self.idx_decomp,
        );

        let mut rows: Vec<Vec<Val<SC>>> = vec![];

        for i in 0..page.len() {
            let prv_idx = if i == 0 {
                vec![0; self.idx_len]
            } else {
                page[i - 1][1..1 + self.idx_len].to_vec()
            };

            let cur_idx = page[i][1..1 + self.idx_len].to_vec();

            let lt_cols: IsLessThanTupleCols<Val<SC>> = LocalTraceInstructions::generate_trace_row(
                &lt_chip,
                (prv_idx, cur_idx, range_checker.clone()),
            );

            let page_aux_cols = FinalPageAuxCols {
                lt_cols: lt_cols.aux,
                lt_out: lt_cols.io.tuple_less_than,
            };

            rows.push(page_aux_cols.flatten());
        }

        RowMajorMatrix::new(rows.concat(), self.aux_width())
    }
}
