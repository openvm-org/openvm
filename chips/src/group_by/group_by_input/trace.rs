use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

use crate::sub_chip::LocalTraceInstructions;

use super::GroupByAir;

impl GroupByAir {
    pub fn gen_page_trace<F: Field>(&self, page: Vec<Vec<u32>>) -> RowMajorMatrix<F> {
        RowMajorMatrix::new(
            page.into_iter()
                .flat_map(|row| {
                    row.into_iter()
                        .map(F::from_canonical_u32)
                        .collect::<Vec<F>>()
                })
                .collect(),
            self.page_width,
        )
    }

    pub fn gen_aux_trace<F: Field>(&self, page: Vec<Vec<u32>>) -> RowMajorMatrix<F> {
        let page_f: Vec<Vec<F>> = page
            .iter()
            .map(|row| row.iter().map(|&x| F::from_canonical_u32(x)).collect())
            .collect();

        let mut grouped_page: Vec<Vec<u32>> = page
            .iter()
            .map(|row| {
                let mut selected_row: Vec<u32> = self
                    .group_by_cols
                    .iter()
                    .map(|&col_index| row[col_index])
                    .collect();
                selected_row.push(row[self.aggregated_col]);
                selected_row.insert(0, 1);
                selected_row
            })
            .collect();

        grouped_page.sort();

        let grouped_page: Vec<Vec<F>> = grouped_page
            .iter()
            .map(|row| row.iter().map(|&x| F::from_canonical_u32(x)).collect())
            .collect();

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

        let is_final = is_equal
            .iter()
            .map(|row| vec![F::one() - row[0]])
            .collect::<Vec<Vec<F>>>();

        let num_group_by_cols = self.group_by_cols.len();
        let mut partial_sums: Vec<Vec<F>> = vec![vec![F::zero()]; grouped_page.len()];
        if !grouped_page.is_empty() {
            partial_sums[0][0] = grouped_page[0][num_group_by_cols + 1]; // Initialize with the first aggregated value
            for i in 1..grouped_page.len() {
                partial_sums[i][0] = partial_sums[i - 1][0] * is_equal[i - 1][0]
                    + grouped_page[i][num_group_by_cols + 1];
            }
        }

        let mut eq_vec_aux_trace: Vec<Vec<F>> = vec![];
        for pair in page_f.windows(2) {
            let local_is_eq_vec_cols = LocalTraceInstructions::generate_trace_row(
                &self.is_equal_vec_air,
                (pair[0].clone(), pair[1].clone()),
            );
            eq_vec_aux_trace.push(local_is_eq_vec_cols.aux.flatten());
        }
        eq_vec_aux_trace.push(vec![F::zero(); self.is_equal_vec_air.aux_width()]);

        let mut trace: Vec<F> = grouped_page
            .iter()
            .zip(partial_sums.iter())
            .zip(is_final.iter())
            .zip(is_equal.iter())
            .zip(eq_vec_aux_trace.iter())
            .flat_map(
                |((((grouped_row, partial_sum_row), is_final_row), is_eq_row), eq_vec_aux_row)| {
                    let mut trace_row = grouped_row.clone();
                    trace_row.extend(partial_sum_row.clone()); // Singleton from partial
                    trace_row.extend(is_final_row.clone());
                    trace_row.extend(is_eq_row.clone());
                    trace_row.extend(eq_vec_aux_row.clone());

                    trace_row.into_iter()
                },
            )
            .collect();

        let num_rows = page.len();
        let target_length = num_rows.next_power_of_two();
        let additional_cols = target_length - num_rows;
        trace.extend(vec![F::zero(); additional_cols * self.aux_width()]);

        RowMajorMatrix::new(trace, self.aux_width())
    }

    // pub fn generate_trace<F: Field>(&self, page: Vec<Vec<u32>>) -> RowMajorMatrix<F> {
    //     let page_trace = self.gen_page_trace(page);
    //     let aux_trace = self.gen_aux_trace(page);

    //     // RowMajorMatrix::new(trace, self.get_width())
    // }
}
