use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use crate::sub_chip::LocalTraceInstructions;

use super::GroupByChip;

impl GroupByChip {
    pub fn generate_trace<F: PrimeField64>(&self, page: Vec<Vec<F>>) -> RowMajorMatrix<F> {
        let mut grouped_page: Vec<Vec<F>> = page
            .iter()
            .map(|row| {
                let mut selected_row: Vec<F> = self
                    .air
                    .group_by_cols
                    .iter()
                    .map(|&col_index| row[col_index])
                    .collect();
                selected_row.push(row[self.air.aggregated_col]);
                selected_row
            })
            .collect();

        grouped_page.sort();

        let mut is_equal: Vec<Vec<F>> = grouped_page
            .windows(2)
            .map(|pair| {
                vec![if pair[0] == pair[1] {
                    F::one()
                } else {
                    F::zero()
                }]
            })
            .collect();
        is_equal.push(vec![F::zero()]);

        let num_group_by_cols = self.air.group_by_cols.len();
        let mut partial_sums: Vec<Vec<F>> = vec![vec![F::zero()]; grouped_page.len()];
        if !grouped_page.is_empty() {
            partial_sums[0][0] = grouped_page[0][num_group_by_cols]; // Initialize with the first aggregated value
            for i in 1..grouped_page.len() {
                partial_sums[i][0] = partial_sums[i - 1][0] * is_equal[i - 1][0]
                    + grouped_page[i][num_group_by_cols];
            }
        }

        let mut eq_vec_aux_trace: Vec<Vec<F>> = vec![];
        for pair in page.windows(2) {
            let local_is_eq_vec_cols = LocalTraceInstructions::generate_trace_row(
                &self.air.is_equal_vec_air,
                (pair[0].clone(), pair[1].clone()),
            );
            eq_vec_aux_trace.push(local_is_eq_vec_cols.aux.flatten());
        }

        let trace = page
            .iter()
            .zip(grouped_page.iter())
            .zip(partial_sums.iter())
            .zip(is_equal.iter())
            .zip(eq_vec_aux_trace.iter())
            .flat_map(
                |((((page_row, grouped_row), partial_sum_row), is_eq_row), eq_vec_aux_row)| {
                    let mut trace_row = page_row.clone();
                    trace_row.extend(grouped_row.clone());
                    trace_row.extend(partial_sum_row.clone()); // Singleton from partial
                    trace_row.extend(is_eq_row.clone());
                    trace_row.extend(eq_vec_aux_row.clone());

                    trace_row.into_iter()
                },
            )
            .collect();

        RowMajorMatrix::new(trace, self.air.get_width())
    }
}
