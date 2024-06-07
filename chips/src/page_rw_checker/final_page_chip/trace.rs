use std::{iter, sync::Arc};

use p3_field::{AbstractField, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};

use super::FinalPageChip;
use crate::{
    is_less_than_tuple::{columns::IsLessThanTupleCols, IsLessThanTupleAir},
    range_gate::RangeCheckerGateChip,
    sub_chip::LocalTraceInstructions,
};

impl FinalPageChip {
    // The trace is the whole page (including the is_alloc column)
    pub fn gen_page_trace<SC: StarkGenericConfig>(
        &self,
        page: Vec<Vec<u32>>,
    ) -> RowMajorMatrix<Val<SC>>
    where
        Val<SC>: AbstractField,
    {
        println!(
            "in final page trace: {} {} {}",
            page.len(),
            page[0].len(),
            self.page_width()
        );

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

    pub fn gen_aux_trace<SC: StarkGenericConfig>(
        &self,
        page: Vec<Vec<u32>>,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> RowMajorMatrix<Val<SC>>
    where
        Val<SC>: PrimeField64,
    {
        let lt_chip = IsLessThanTupleAir::new(
            self.sorted_bus_index,
            1 << self.idx_limb_bits,
            vec![self.idx_limb_bits; 1 + self.idx_len],
            self.idx_decomp,
        );

        let mut rows: Vec<Vec<Val<SC>>> = vec![vec![Val::<SC>::zero(); self.aux_width()]];

        for i in 1..page.len() {
            let mut prv_r = page[i - 1][0..1 + self.idx_len].to_vec();
            let mut cur_r = page[i][0..1 + self.idx_len].to_vec();

            println!("ok here: {} {}", prv_r[0], cur_r[0]);

            prv_r[0] = 1 - prv_r[0];
            cur_r[0] = 1 - cur_r[0];

            let cols: IsLessThanTupleCols<Val<SC>> = LocalTraceInstructions::generate_trace_row(
                &lt_chip,
                (prv_r, cur_r, range_checker.clone()),
            );

            rows.push(
                cols.aux
                    .flatten()
                    .iter()
                    .chain(iter::once(&cols.io.tuple_less_than))
                    .cloned()
                    .collect(),
            );
        }

        RowMajorMatrix::new(rows.concat(), self.aux_width())
    }
}
