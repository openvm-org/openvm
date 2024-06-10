use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use crate::sub_chip::LocalTraceInstructions;

use super::{columns::PageIndexScanInputCols, PageIndexScanInputChip};

impl PageIndexScanInputChip {
    pub fn generate_trace<F: PrimeField64>(
        &self,
        page: Vec<Vec<u32>>,
        x: Vec<u32>,
    ) -> RowMajorMatrix<F> {
        let num_cols: usize = PageIndexScanInputCols::<F>::get_width(
            self.air.idx_len,
            self.air.data_len,
            self.air.is_less_than_tuple_air.limb_bits(),
            self.air.is_less_than_tuple_air.decomp(),
        );

        let mut rows: Vec<F> = vec![];

        for page_row in &page {
            let mut row: Vec<F> = vec![];

            let is_alloc = F::from_canonical_u32(page_row[0]);
            row.push(is_alloc);

            let idx = page_row[1..1 + self.air.idx_len].to_vec();
            let idx_trace: Vec<F> = idx.iter().map(|x| F::from_canonical_u32(*x)).collect();
            row.extend(idx_trace);

            let data =
                page_row[1 + self.air.idx_len..1 + self.air.idx_len + self.air.data_len].to_vec();
            let data_trace: Vec<F> = data.iter().map(|x| F::from_canonical_u32(*x)).collect();
            row.extend(data_trace);

            let x_trace: Vec<F> = x.iter().map(|x| F::from_canonical_u32(*x)).collect();
            row.extend(x_trace);

            let is_less_than_tuple_trace: Vec<F> = LocalTraceInstructions::generate_trace_row(
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

        RowMajorMatrix::new(rows, num_cols)
    }
}
