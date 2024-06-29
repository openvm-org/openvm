use std::iter;
use std::sync::Arc;

use afs_test_utils::utils::to_field_vec;
use p3_field::{AbstractField, PrimeField};
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};

use super::columns::OfflineCheckerCols;
use super::OfflineChecker;
use crate::common::page::Page;
use crate::page_rw_checker::page_controller::{OpType, Operation};
use crate::range_gate::RangeCheckerGateChip;
use crate::sub_chip::LocalTraceInstructions;

impl OfflineChecker {
    /// Each row in the trace follow the same order as the Cols struct:
    /// [is_initial, is_final_write, is_final_delete, is_internal, is_final_write_x3, clk, idx, data, op_type, same_idx, lt_bit, is_extra, is_equal_idx_aux, lt_aux]
    ///
    /// The trace consists of a row for every read/write/delete operation plus some extra rows
    /// The trace is sorted by idx and then by clk, so every idx has a block of consective rows in the trace with the following structure
    /// If the index exists in the initial page, the block starts with a row of the initial data with is_initial=1
    /// Then, a row is added to the trace for every read/write/delete operation with the corresponding data with is_internal=1
    /// Then, a row is added with the final data (or vector of zeros if deleted) for that index with is_final_write=1 or is_final_delete=1
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
                       last_clk: &mut usize,
                       is_extra: bool| {
            if *is_first_row {
                // Making sure the last_idx and last_data are different from current when its the first row
                last_idx.clone_from(cur_idx);

                last_idx[0] += 1;

                *is_first_row = false;
            }

            let my_last_idx = last_idx.clone();
            let my_last_clk = *last_clk;

            last_idx.clone_from(cur_idx);
            *last_clk = clk;

            let last_idx = my_last_idx;
            let last_clk = my_last_clk;

            let lt_cols = LocalTraceInstructions::generate_trace_row(
                &self.lt_idx_clk_air,
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

            // those are used later to initialize the Cols struct
            let my_cur_idx = cur_idx.clone();
            let my_cur_data = cur_data.clone();

            let last_idx = to_field_vec(last_idx);
            let cur_idx = to_field_vec(cur_idx.to_vec());

            let is_equal_idx_cols = LocalTraceInstructions::generate_trace_row(
                &self.is_equal_idx_air,
                (last_idx, cur_idx),
            );

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
                to_field_vec(my_cur_idx),
                to_field_vec(my_cur_data),
                Val::<SC>::from_canonical_u8(op_type),
                Val::<SC>::from_bool(is_read),
                Val::<SC>::from_bool(is_write),
                Val::<SC>::from_bool(is_delete),
                is_equal_idx_cols.io.prod,
                lt_cols.io.tuple_less_than,
                Val::<SC>::from_bool(is_extra),
                is_equal_idx_cols.aux,
                lt_cols.aux,
            );

            assert!(cols.flatten().len() == self.air_width());
            cols.flatten()
        };

        let mut rows = vec![];

        let mut last_idx = vec![0; self.idx_len];
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
                    op.op_type as u8,
                    &mut last_idx,
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
                &mut last_clk,
                true,
            )
        });

        tracing::debug_span!("Offline Checker trace by row: ").in_scope(|| {
            for row in &rows {
                let cols = OfflineCheckerCols::from_slice(row, self);
                tracing::debug!("{:?}", cols);
            }
        });

        RowMajorMatrix::new(rows.concat(), self.air_width())
    }
}
