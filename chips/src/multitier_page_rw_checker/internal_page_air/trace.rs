use std::sync::Arc;

use itertools::Itertools;
use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    is_less_than_tuple::IsLessThanTupleAir, range_gate::RangeCheckerGateChip,
    sub_chip::LocalTraceInstructions,
};

use super::InternalPageAir;

impl<const COMMITMENT_LEN: usize> InternalPageAir<COMMITMENT_LEN> {
    // The cached trace is the whole page (including the is_leaf and is_alloc column)
    pub fn generate_cached_trace<F: PrimeField64>(&self, page: Vec<Vec<u32>>) -> RowMajorMatrix<F> {
        RowMajorMatrix::new(
            page.into_iter()
                .flat_map(|row| row.into_iter().map(F::from_wrapped_u32).collect::<Vec<F>>())
                .collect(),
            2 + 2 * self.idx_len + COMMITMENT_LEN,
        )
    }

    pub fn generate_main_trace<F: PrimeField64>(
        &self,
        page: Vec<Vec<u32>>,
        commit: Vec<u32>,
        child_ids: Vec<u32>,
        mults: Vec<u32>,
        range: (Vec<u32>, Vec<u32>),
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> RowMajorMatrix<F> {
        assert!(commit.len() == COMMITMENT_LEN);
        RowMajorMatrix::new(
            page.iter()
                .zip(mults)
                .enumerate()
                .flat_map(|(i, (row, mult))| {
                    let mut trace_row = vec![];
                    trace_row.extend(commit.clone());
                    trace_row.push(self.air_id);
                    trace_row.push(child_ids[i]);
                    trace_row.push(mult);
                    trace_row.push(mult * row[1]);
                    trace_row.push((mult * row[1] == 1) as u32);
                    // dummy value
                    trace_row.push(0);
                    trace_row.push(row[1] * mult - row[1]);
                    let next = if i < page.len() - 1 {
                        page[i + 1][2..2 + self.idx_len].to_vec()
                    } else {
                        page[0][2..2 + self.idx_len].to_vec()
                    };
                    if !self.is_init {
                        trace_row.push(1);
                        trace_row.push(0);
                        trace_row.extend(range.0.clone());
                        trace_row.extend(range.1.clone());
                        trace_row.extend(vec![0; 2]);
                        let mut trace_row: Vec<F> = trace_row
                            .into_iter()
                            .map(|i| F::from_wrapped_u32(i))
                            .collect();
                        let mut gen_aux =
                            |idx1: Vec<u32>,
                             idx2: Vec<u32>,
                             lt_res_idx: usize,
                             air: IsLessThanTupleAir| {
                                let lt_cols =
                                    air.generate_trace_row((idx1, idx2, range_checker.clone()));
                                trace_row.extend(lt_cols.aux.flatten());
                                trace_row[COMMITMENT_LEN + 7 + lt_res_idx] =
                                    lt_cols.io.tuple_less_than;
                            };
                        gen_aux(
                            row[2..2 + self.idx_len].to_vec(),
                            range.0.clone(),
                            2 + 2 * self.idx_len,
                            self.is_less_than_tuple_air.clone().unwrap().idx1_start,
                        );
                        gen_aux(
                            range.1.clone(),
                            row[2 + self.idx_len..2 + 2 * self.idx_len].to_vec(),
                            2 + 2 * self.idx_len + 1,
                            self.is_less_than_tuple_air.clone().unwrap().end_idx2,
                        );
                        gen_aux(
                            row[2 + self.idx_len..2 + 2 * self.idx_len].to_vec(),
                            next.clone(),
                            0,
                            self.is_less_than_tuple_air.clone().unwrap().idx2_next,
                        );
                        gen_aux(
                            row[2 + self.idx_len..2 + 2 * self.idx_len].to_vec(),
                            row[2..2 + self.idx_len].to_vec(),
                            1,
                            self.is_less_than_tuple_air.clone().unwrap().idx2_idx1,
                        );
                        trace_row.push(
                            self.is_less_than_tuple_air
                                .clone()
                                .unwrap()
                                .mult_is_1
                                .generate_trace_row(F::from_wrapped_u32(mult * row[1]) - F::one())
                                .inv,
                        );
                        trace_row[COMMITMENT_LEN + 5] = trace_row[COMMITMENT_LEN + 3] - F::one();
                        trace_row
                    } else {
                        let mut trace_row = trace_row
                            .into_iter()
                            .map(|i| F::from_wrapped_u32(i))
                            .collect_vec();
                        trace_row[COMMITMENT_LEN + 5] = trace_row[COMMITMENT_LEN + 3] - F::one();
                        trace_row
                    }
                })
                .collect(),
            self.air_width() - (2 + 2 * self.idx_len + COMMITMENT_LEN),
        )
    }
}
