use crate::common::page::Page;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

use crate::sub_chip::LocalTraceInstructions;

use super::{columns::GroupByCols, GroupByAir};

impl GroupByAir {
    pub fn gen_aux_trace<F: Field>(&self, grouped_page: &Page) -> RowMajorMatrix<F> {
        let page_f: Vec<Vec<F>> = grouped_page
            .rows
            .iter()
            .map(|row| {
                row.to_vec()
                    .iter()
                    .map(|&x| F::from_canonical_u32(x))
                    .collect()
            })
            .collect();

        let mut is_equal: Vec<Vec<F>> = grouped_page
            .rows
            .windows(2)
            .map(|pair| {
                vec![if pair[0].idx == pair[1].idx {
                    F::one()
                } else {
                    F::zero()
                }]
            })
            .collect();
        is_equal.push(vec![F::one()]);

        let is_final = is_equal
            .iter()
            .map(|row| vec![F::one() - row[0]])
            .collect::<Vec<Vec<F>>>();

        let index_cols_map = GroupByCols::<u32>::index_map(self);
        let agg_idx = index_cols_map.aggregated - index_cols_map.page_range.end;
        let mut partial_sums: Vec<Vec<F>> = vec![vec![F::zero()]; page_f.len()];
        if !page_f.is_empty() {
            partial_sums[0][0] = page_f[0][agg_idx]; // Initialize with the first aggregated value
            for i in 1..page_f.len() {
                partial_sums[i][0] =
                    partial_sums[i - 1][0] * is_equal[i - 1][0] + page_f[i][agg_idx];
            }
        }

        let mut eq_vec_aux_trace: Vec<Vec<F>> = vec![];
        for pair in page_f.windows(2) {
            let vecs: Vec<Vec<F>> = pair
                .iter()
                .map(|row| row[1..1 + self.group_by_cols.len()].to_vec())
                .collect();
            let local_is_eq_vec_cols = LocalTraceInstructions::generate_trace_row(
                &self.is_equal_vec_air,
                (vecs[0].clone(), vecs[1].clone()),
            );
            eq_vec_aux_trace.push(local_is_eq_vec_cols.aux.flatten());
        }
        eq_vec_aux_trace.push(vec![F::zero(); self.is_equal_vec_air.aux_width()]);

        let trace: Vec<F> = page_f
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

        RowMajorMatrix::new(trace, self.aux_width())
    }
}
