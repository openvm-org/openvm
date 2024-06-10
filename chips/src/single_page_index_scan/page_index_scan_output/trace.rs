use p3_field::{AbstractField, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};

use crate::sub_chip::LocalTraceInstructions;

use super::PageIndexScanOutputChip;

impl PageIndexScanOutputChip {
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

    pub fn gen_aux_trace<SC: StarkGenericConfig>(
        &self,
        page: Vec<Vec<u32>>,
    ) -> RowMajorMatrix<Val<SC>>
    where
        Val<SC>: AbstractField + PrimeField64,
    {
        let mut rows: Vec<Val<SC>> = vec![];

        for i in 0..page.len() {
            let page_row = page[i].clone();
            let next_page: Vec<u32> = if i == page.len() - 1 {
                vec![0; 1 + self.air.idx_len + self.air.data_len]
            } else {
                page[i + 1].clone()
            };

            let mut row: Vec<Val<SC>> = vec![];

            let is_less_than_tuple_trace = LocalTraceInstructions::generate_trace_row(
                self.air.is_less_than_tuple_air(),
                (
                    page_row[1..1 + self.air.idx_len].to_vec(),
                    next_page[1..1 + self.air.idx_len].to_vec(),
                    self.range_checker.clone(),
                ),
            )
            .flatten();

            row.extend_from_slice(&is_less_than_tuple_trace[2 * self.air.idx_len..]);

            rows.extend_from_slice(&row);
        }

        RowMajorMatrix::new(rows, self.aux_width())
    }
}
