use std::iter;
use std::sync::Arc;

use afs_test_utils::utils::to_field_vec;
use p3_field::{AbstractField, PrimeField};
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};

use super::columns::OfflineCheckerCols;
use super::OfflineChecker;
use crate::common::page::Page;
use crate::is_equal_vec::IsEqualVecAir;
use crate::is_less_than_tuple::IsLessThanTupleAir;
use crate::page_rw_checker::page_controller::{OpType, Operation};
use crate::range_gate::RangeCheckerGateChip;
use crate::sub_chip::LocalTraceInstructions;

impl OfflineChecker {
    /// Each row in the trace follow the same order as the Cols struct:
    /// [is_initial, is_final, is_internal, is_final_x3, clk, page_row, op_type, same_idx, same_data, lt_bit, is_extra, is_equal_idx_aux, is_equal_data_aux, lt_aux]
    ///
    /// The trace consists of a row for every read/write operation plus some extra rows
    /// The trace is sorted by index (in page_row) and then by clk, so every index has a block of consective rows in the trace with the following structure
    /// If the index exists in the initial page, the block starts with a row of the initial data with is_initial=1
    /// Then, a row is added to the trace for every read/write operation with the corresponding data with is_internal=1
    /// Then, a row is added with the final data for that index with is_final=1
    /// The trace is padded at the end to be of height trace_degree
    pub fn generate_trace<SC: StarkGenericConfig>(
        &self,
        page: &mut Page,
        mut ops: Vec<Operation>,
        range_checker: Arc<RangeCheckerGateChip>,
        trace_degree: usize,
    ) -> RowMajorMatrix<Val<SC>>
    where
        Val<SC>: PrimeField,
    {
        let is_equal_idx = IsEqualVecAir::new(self.idx_len);
        let is_equal_data = IsEqualVecAir::new(self.data_len);

        let lt_chip =
            IsLessThanTupleAir::new(usize::MAX, self.idx_clk_limb_bits.clone(), self.idx_decomp);

        // Creating a timestamp bigger than all others
        let max_clk = ops.iter().map(|op| op.clk).max().unwrap_or(0) + 1;

        ops.sort_by_key(|op| (op.idx.clone(), op.clk));

        // This takes the information for the current row and references for the last row
        // It uses those values to generate the new row in the trace, and it updates the references
        // to the new row's information
        let gen_row = |is_first_row: &mut bool,
                       cur_idx: &Vec<u32>,
                       cur_data: &Vec<u32>,
                       is_initial: bool,
                       is_final: bool,
                       is_internal: bool,
                       clk: usize,
                       op_type: u8,
                       last_idx: &mut Vec<u32>,
                       last_data: &mut Vec<u32>,
                       last_clk: &mut usize,
                       is_extra: bool| {
            if *is_first_row {
                // Making sure the last_idx and last_data are different from current when its the first row
                last_idx.clone_from(cur_idx);
                last_data.clone_from(cur_data);

                last_idx[0] += 1;
                last_data[0] += 1;

                *is_first_row = false;
            }

            let my_last_idx = last_idx.clone();
            let my_last_data = last_data.clone();
            let my_last_clk = *last_clk;

            last_idx.clone_from(cur_idx);
            last_data.clone_from(cur_data);
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

            let page_row: Vec<u32> = cur_idx.iter().chain(cur_data.iter()).copied().collect();

            let last_idx = to_field_vec(last_idx);
            let cur_idx = to_field_vec(cur_idx.to_vec());

            let last_data = to_field_vec(last_data);
            let cur_data = to_field_vec(cur_data.to_vec());

            let idx_equal_cols =
                LocalTraceInstructions::generate_trace_row(&is_equal_idx, (last_idx, cur_idx));

            let data_equal_cols =
                LocalTraceInstructions::generate_trace_row(&is_equal_data, (last_data, cur_data));

            let is_final_write = is_final && op_type == 0;
            let is_final_delete = is_final && op_type == 2;
            let is_read = op_type == 0;
            let is_write = op_type == 1;
            let is_delete = op_type == 2;

            let cols = OfflineCheckerCols::new(
                Val::<SC>::from_bool(is_initial),
                Val::<SC>::from_bool(is_final_write),
                Val::<SC>::from_bool(is_final_delete),
                Val::<SC>::from_bool(is_internal),
                Val::<SC>::from_canonical_u8(is_final_write as u8 * 3),
                Val::<SC>::from_canonical_usize(clk),
                page_row
                    .into_iter()
                    .map(Val::<SC>::from_canonical_u32)
                    .collect(),
                Val::<SC>::from_canonical_u8(op_type),
                Val::<SC>::from_bool(is_read),
                Val::<SC>::from_bool(is_write),
                Val::<SC>::from_bool(is_delete),
                idx_equal_cols.io.prod,
                data_equal_cols.io.prod,
                lt_cols.io.tuple_less_than,
                Val::<SC>::from_bool(is_extra),
                idx_equal_cols.aux,
                data_equal_cols.aux,
                lt_cols.aux,
            );

            assert!(cols.flatten().len() == self.air_width());
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

            if page.contains(&cur_idx) {
                // Adding the is_initial row to the trace
                rows.push(gen_row(
                    &mut is_first_row,
                    &cur_idx,
                    &page[&cur_idx],
                    true,
                    false,
                    false,
                    0,
                    1,
                    &mut last_idx,
                    &mut last_data,
                    &mut last_clk,
                    false,
                ));
            }

            for op in ops.iter().take(j).skip(i) {
                if op.op_type == OpType::Write {
                    if !page.contains(&cur_idx) {
                        page.insert(&cur_idx, &op.data);
                    } else {
                        page[&cur_idx].clone_from(&op.data);
                    }
                } else if op.op_type == OpType::Delete {
                    page.delete(&cur_idx);
                }

                rows.push(gen_row(
                    &mut is_first_row,
                    &cur_idx,
                    &op.data,
                    false,
                    false,
                    true,
                    op.clk,
                    op.op_type.clone() as u8,
                    &mut last_idx,
                    &mut last_data,
                    &mut last_clk,
                    false,
                ));
            }

            let final_data = if page.contains(&cur_idx) {
                &page[&cur_idx]
            } else {
                &vec![0; self.data_len]
            };

            // Adding the is_final row to the trace
            rows.push(gen_row(
                &mut is_first_row,
                &cur_idx,
                final_data,
                false,
                true,
                false,
                max_clk,
                if page.contains(&cur_idx) { 0 } else { 2 }, // 0 (read) for is_final_write, 2 (delete) for is_final_delete
                &mut last_idx,
                &mut last_data,
                &mut last_clk,
                false,
            ));

            i = j;
        }

        // Ensure that trace degree is a power of two
        assert!(trace_degree > 0 && trace_degree & (trace_degree - 1) == 0);

        // dummy idx
        let idx = page[0].idx.clone();

        // Adding rows to the trace to make the height trace_degree
        rows.resize_with(trace_degree, || {
            gen_row(
                &mut is_first_row,
                &idx,
                &vec![0; self.data_len],
                false,
                false,
                false,
                0,
                0,
                &mut last_idx,
                &mut last_data,
                &mut last_clk,
                true,
            )
        });

        tracing::debug!("Offline Checker trace by row: ");
        for row in &rows {
            let cols = OfflineCheckerCols::from_slice(row, self);
            tracing::debug!("{:?}", cols);
        }

        RowMajorMatrix::new(rows.concat(), self.air_width())
    }
}
