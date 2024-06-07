use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use super::{columns::PageIndexScanVerifyCols, PageIndexScanVerifyChip};

impl PageIndexScanVerifyChip {
    pub fn generate_trace<F: PrimeField64>(&self, page: Vec<Vec<u32>>) -> RowMajorMatrix<F> {
        let num_cols: usize =
            PageIndexScanVerifyCols::<F>::get_width(self.air.idx_len, self.air.data_len);

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

            rows.extend_from_slice(&row);
        }

        RowMajorMatrix::new(rows, num_cols)
    }
}
