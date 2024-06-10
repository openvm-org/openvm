use std::sync::Arc;
use std::{collections::HashMap, iter};

use p3_field::{AbstractField, PrimeField};
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};

use super::columns::OfflineCheckerCols;
use super::OfflineChecker;
use crate::is_equal_vec::IsEqualVecAir;
use crate::is_less_than_tuple::IsLessThanTupleAir;
use crate::page_rw_checker::page_controller::Operation;
use crate::range_gate::RangeCheckerGateChip;
use crate::sub_chip::LocalTraceInstructions;

impl OfflineChecker {
    // Each row in the trace follow the same order as the Cols struct:
    // [is_initial, is_final, clk, page_row, op_type, same_idx, same_data, is_extra, is_equal_idx_aux, is_equal_data_aux]
    //
    // The trace consists of a row for every read/write operation plus some extra rows
    // The trace is sorted by index (in page_row) and then by clk, so every index has a block of consective rows in the trace with the following structure
    // If the index exists in the initial page, the block starts with a row of the initial data with is_initial=1
    // Then, a row is added to the trace for every read/write operation with the corresponding data
    // Then, a row is added with the final data for that index with is_final=1
    // The trace is padded at the end to be of height trace_degree
    pub fn generate_trace<SC: StarkGenericConfig>(
        &self,
        page: &mut Vec<Vec<u32>>,
        mut ops: Vec<Operation>,
        range_checker: Arc<RangeCheckerGateChip>,
        trace_degree: usize,
    ) -> RowMajorMatrix<Val<SC>>
    where
        Val<SC>: PrimeField,
    {
        let is_equal_idx = IsEqualVecAir::new(self.idx_len);
        let is_equal_data = IsEqualVecAir::new(self.data_len);

        let lt_chip = IsLessThanTupleAir::new(
            usize::MAX,
            1 << self.idx_decomp,
            self.idx_clk_limb_bits.clone(),
            self.idx_decomp,
        );

        let mut rows_allocated = 0;
        while rows_allocated < page.len() && page[rows_allocated][0] == 1 {
            rows_allocated += 1;
        }

        let mut idx_i_map = HashMap::new();
        for (i, row) in page.iter().enumerate().take(rows_allocated) {
            idx_i_map.insert(row[1..self.idx_len + 1].to_vec(), i);
        }

        // Creating a timestamp bigger than all others
        let max_clk = ops.iter().map(|op| op.clk).max().unwrap_or(0) + 1;

        ops.sort_by_key(|op| (op.idx.clone(), op.clk));

        let conv_to_f = |v: Vec<u32>| {
            v.iter()
                .map(|x| Val::<SC>::from_canonical_u32(*x))
                .collect::<Vec<Val<SC>>>()
        };

        let gen_row = |is_first_row: &mut bool,
                       page: &mut Vec<Vec<u32>>,
                       idx: usize,
                       is_initial: u8,
                       is_final: u8,
                       is_internal: u8,
                       clk: usize,
                       op_type: u8,
                       last_idx: &mut Vec<u32>,
                       last_data: &mut Vec<u32>,
                       last_clk: &mut usize,
                       is_extra: u8| {
            // Make sure the row in the page is allocated
            assert!(page[idx][0] == 1);

            let cur_idx = page[idx][1..self.idx_len + 1].to_vec();
            let cur_data = page[idx][self.idx_len + 1..].to_vec();

            if *is_first_row {
                // Making sure the last_idx and last_data are different from current when its the first row
                last_idx.clone_from(&cur_idx);
                last_data.clone_from(&cur_data);

                last_idx[0] += 1;
                last_data[0] += 1;

                *is_first_row = false;
            }

            let my_last_idx = last_idx.clone();
            let my_last_data = last_data.clone();
            let my_last_clk = *last_clk;

            last_idx.clone_from(&cur_idx);
            last_data.clone_from(&cur_data);
            *last_clk = clk;

            let last_idx = my_last_idx;
            let last_data = my_last_data;
            let last_clk = my_last_clk;

            let lt_cols = LocalTraceInstructions::generate_trace_row(
                &lt_chip,
                (
                    last_idx
                        .iter()
                        .copied()
                        .chain(iter::once(last_clk as u32))
                        .collect(),
                    cur_idx
                        .iter()
                        .copied()
                        .chain(iter::once(clk as u32))
                        .collect(),
                    range_checker.clone(),
                ),
            );

            // let same_idx = cur_idx == last_idx;
            // let same_data = cur_data == last_data;

            let last_idx = conv_to_f(last_idx);
            let cur_idx = conv_to_f(cur_idx);

            let last_data = conv_to_f(last_data);
            let cur_data = conv_to_f(cur_data);

            let idx_equal_cols =
                LocalTraceInstructions::generate_trace_row(&is_equal_idx, (last_idx, cur_idx));

            let data_equal_cols =
                LocalTraceInstructions::generate_trace_row(&is_equal_data, (last_data, cur_data));

            let cols = OfflineCheckerCols::new(
                Val::<SC>::from_canonical_u8(is_initial),
                Val::<SC>::from_canonical_u8(is_final),
                Val::<SC>::from_canonical_u8(is_internal),
                Val::<SC>::from_canonical_usize(clk),
                page[idx]
                    .iter()
                    .copied()
                    .map(Val::<SC>::from_canonical_u32)
                    .collect(),
                Val::<SC>::from_canonical_u8(op_type),
                idx_equal_cols.io.prod,
                data_equal_cols.io.prod,
                lt_cols.io.tuple_less_than,
                Val::<SC>::from_canonical_u8(is_extra),
                idx_equal_cols.aux,
                data_equal_cols.aux,
                lt_cols.aux,
            );

            cols.flatten()
        };

        let mut rows = vec![];

        let mut last_idx = vec![0; self.idx_len];
        let mut last_data = vec![0; self.data_len];
        let mut last_clk = 0;
        let mut is_first_row = true;

        let mut i = 0;
        while i < ops.len() {
            let cur_idx = ops[i].idx.clone();

            let mut j = i + 1;
            while j < ops.len() && ops[j].idx == cur_idx {
                j += 1;
            }

            let idx;
            if let std::collections::hash_map::Entry::Vacant(e) = idx_i_map.entry(cur_idx.clone()) {
                assert!(rows_allocated < page.len());
                idx = rows_allocated;
                e.insert(idx);
                rows_allocated += 1;
            } else {
                // Adding the is_initial row to the trace
                idx = *idx_i_map.get(&cur_idx).unwrap();

                rows.push(gen_row(
                    &mut is_first_row,
                    page,
                    idx,
                    1,
                    0,
                    0,
                    0,
                    1,
                    &mut last_idx,
                    &mut last_data,
                    &mut last_clk,
                    0,
                ));
            }

            for op in ops.iter().take(j).skip(i) {
                page[idx] = iter::once(1)
                    .chain(op.idx.clone())
                    .chain(op.data.clone())
                    .collect();

                rows.push(gen_row(
                    &mut is_first_row,
                    page,
                    idx,
                    0,
                    0,
                    1,
                    op.clk,
                    op.op_type.clone() as u8,
                    &mut last_idx,
                    &mut last_data,
                    &mut last_clk,
                    0,
                ));
            }

            // Adding the is_final row to the trace
            rows.push(gen_row(
                &mut is_first_row,
                page,
                idx,
                0,
                1,
                0,
                max_clk,
                0,
                &mut last_idx,
                &mut last_data,
                &mut last_clk,
                0,
            ));

            i = j;
        }

        // Ensure that trace degree is a power of two
        assert!(trace_degree > 0 && trace_degree & (trace_degree - 1) == 0);

        // Adding rows to the trace to make the height trace_degree
        while rows.len() < trace_degree {
            rows.push(gen_row(
                &mut is_first_row,
                page,
                0,
                0,
                0,
                0,
                0,
                0,
                &mut last_idx,
                &mut last_data,
                &mut last_clk,
                1,
            ));
        }

        RowMajorMatrix::new(rows.concat(), self.air_width())
    }
}
