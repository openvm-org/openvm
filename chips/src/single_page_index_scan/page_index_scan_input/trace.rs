use p3_field::{AbstractField, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};

use crate::sub_chip::LocalTraceInstructions;

use super::PageIndexScanInputChip;

impl PageIndexScanInputChip {
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
        x: Vec<u32>,
    ) -> RowMajorMatrix<Val<SC>>
    where
        Val<SC>: AbstractField + PrimeField64,
    {
        let mut rows: Vec<Val<SC>> = vec![];

        for page_row in &page {
            let mut row: Vec<Val<SC>> = vec![];

            let is_alloc = Val::<SC>::from_canonical_u32(page_row[0]);
            let idx = page_row[1..1 + self.air.idx_len].to_vec();

            let x_trace: Vec<Val<SC>> = x
                .iter()
                .map(|x| Val::<SC>::from_canonical_u32(*x))
                .collect();
            row.extend(x_trace);

            let is_less_than_tuple_trace: Vec<Val<SC>> =
                LocalTraceInstructions::generate_trace_row(
                    &self.air.is_less_than_tuple_air,
                    (idx.clone(), x.clone(), self.range_checker.clone()),
                )
                .flatten();

            row.push(is_less_than_tuple_trace[2 * self.air.idx_len]);
            let send_row = is_less_than_tuple_trace[2 * self.air.idx_len] * is_alloc;
            row.push(send_row);

            row.extend_from_slice(&is_less_than_tuple_trace[2 * self.air.idx_len + 1..]);

            rows.extend_from_slice(&row);
        }

        RowMajorMatrix::new(rows, self.aux_width())
    }

    pub fn gen_output(&self, page: Vec<Vec<u32>>, x: Vec<u32>) -> Vec<Vec<u32>> {
        let mut output: Vec<Vec<u32>> = vec![];

        for page_row in &page {
            let is_alloc = page_row[0];
            let idx = page_row[1..1 + self.air.idx_len].to_vec();
            let data = page_row[1 + self.air.idx_len..].to_vec();

            let mut less_than = false;
            for (&idx_val, &x_val) in idx.iter().zip(x.iter()) {
                use std::cmp::Ordering;
                match idx_val.cmp(&x_val) {
                    Ordering::Less => {
                        less_than = true;
                        break;
                    }
                    Ordering::Greater => {
                        break;
                    }
                    Ordering::Equal => {}
                }
            }

            if less_than {
                output.push(
                    vec![is_alloc]
                        .into_iter()
                        .chain(idx.iter().cloned())
                        .chain(data.iter().cloned())
                        .collect(),
                );
            }
        }

        let num_remaining = page.len() - output.len();

        output.extend((0..num_remaining).map(|_| vec![0; self.page_width()]));

        output
    }
}
