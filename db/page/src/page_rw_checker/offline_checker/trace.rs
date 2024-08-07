use std::sync::Arc;

use afs_primitives::{offline_checker::OfflineCheckerChip, range_gate::RangeCheckerGateChip};
use p3_field::{AbstractField, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
#[cfg(feature = "parallel")]
use p3_maybe_rayon::prelude::*;
use p3_uni_stark::{StarkGenericConfig, Val};

use super::{columns::PageOfflineCheckerColsMut, PageOfflineChecker};
use crate::{
    common::{indexed_page_editor::IndexedPageEditor, page::Page},
    page_rw_checker::page_controller::{OpType, Operation},
};

impl PageOfflineChecker {
    /// Each row in the trace follow the same order as the Cols struct:
    /// [is_initial, is_final_write, is_final_delete, is_internal, clk, idx, data, op_type, same_idx, lt_bit, is_extra, is_equal_idx_aux, lt_aux]
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
        Val<SC>: PrimeField64,
    {
        let mut page_editor = IndexedPageEditor::from_page(page);

        // Creating a timestamp bigger than all others
        let max_clk = ops.iter().map(|op| op.clk).max().unwrap_or(0) + 1;

        #[cfg(feature = "parallel")]
        ops.par_sort_by(|a, b| a.idx.cmp(&b.idx).then_with(|| a.clk.cmp(&b.clk)));
        #[cfg(not(feature = "parallel"))]
        ops.sort_by(|a, b| a.idx.cmp(&b.idx).then_with(|| a.clk.cmp(&b.clk)));

        let dummy_op = Operation {
            idx: vec![0; self.offline_checker.idx_len],
            data: vec![0; self.offline_checker.data_len],
            op_type: OpType::Read,
            clk: 0,
        };

        // This takes the operations for the previous row and current row and some extra information.
        // It uses those values to generate the new row in the trace
        let gen_row = |slc: &mut [Val<SC>],
                       is_first_row: &mut bool,
                       is_initial: bool,
                       is_final: bool,
                       is_internal: bool,
                       curr_op: &Operation,
                       prev_op: &Operation,
                       is_valid: bool| {
            let mut oc_cols = PageOfflineCheckerColsMut::from_slice(slc, self);

            self.generate_trace_row(
                is_first_row,
                is_initial,
                is_final,
                is_internal,
                curr_op,
                prev_op,
                is_valid,
                range_checker.clone(),
                &mut oc_cols,
            );
        };

        let width = self.air_width();
        let mut rows_concat = vec![Val::<SC>::zero(); trace_degree * width];
        let mut slc_id = 0;

        let mut is_first_row = true;

        let mut i = 0;
        let mut curr_op = dummy_op.clone();
        let mut prev_op;

        while i < ops.len() {
            let cur_idx = ops[i].idx.clone();

            let mut j = i + 1;
            while j < ops.len() && ops[j].idx == cur_idx {
                j += 1;
            }

            if page_editor.contains(&cur_idx) {
                prev_op = curr_op;
                curr_op = Operation {
                    idx: cur_idx.clone(),
                    data: page_editor.get(&cur_idx).unwrap().clone(),
                    op_type: OpType::Write,
                    clk: 0,
                };

                // Adding the is_initial row to the trace
                gen_row(
                    &mut rows_concat[slc_id * width..(slc_id + 1) * width],
                    &mut is_first_row,
                    true,
                    false,
                    false,
                    &curr_op,
                    &prev_op,
                    true,
                );
                slc_id += 1;
            }

            for op in ops.iter().take(j).skip(i) {
                prev_op = curr_op;
                curr_op = op.clone();

                if op.op_type == OpType::Write {
                    page_editor.write(&cur_idx, &op.data);
                } else if op.op_type == OpType::Delete {
                    page_editor.delete(&cur_idx);
                }

                gen_row(
                    &mut rows_concat[slc_id * width..(slc_id + 1) * width],
                    &mut is_first_row,
                    false,
                    false,
                    true,
                    &curr_op,
                    &prev_op,
                    true,
                );
                slc_id += 1;
            }

            let final_data =
                page_editor
                    .get(&cur_idx)
                    .cloned()
                    .unwrap_or(vec![0; self.offline_checker.data_len]);

            prev_op = curr_op;
            curr_op = Operation {
                idx: cur_idx.clone(),
                data: final_data,
                op_type: if page_editor.contains(&cur_idx) {
                    OpType::Read
                } else {
                    OpType::Delete
                },
                clk: max_clk,
            };

            // Adding the is_final row to the trace
            gen_row(
                &mut rows_concat[slc_id * width..(slc_id + 1) * width],
                &mut is_first_row,
                false,
                true,
                false,
                &curr_op,
                &prev_op,
                true,
            );
            slc_id += 1;

            i = j;
        }

        // Ensure that trace degree is a power of two
        assert!(trace_degree > 0 && trace_degree & (trace_degree - 1) == 0);

        *page = page_editor.into_page();

        // Adding rows to the trace to make the height trace_degree
        while slc_id < trace_degree {
            prev_op = curr_op;
            curr_op = dummy_op.clone();

            gen_row(
                &mut rows_concat[slc_id * width..(slc_id + 1) * width],
                &mut is_first_row,
                false,
                false,
                false,
                &curr_op,
                &prev_op,
                false,
            );
            slc_id += 1;
        }

        RowMajorMatrix::new(rows_concat, width)
    }

    // This takes the operations for the previous row and current row and some extra information.
    // It uses those values to generate the new row in the trace
    #[allow(clippy::too_many_arguments)]
    pub fn generate_trace_row<F: PrimeField64>(
        &self,
        is_first_row: &mut bool,
        is_initial: bool,
        is_final: bool,
        is_internal: bool,
        curr_op: &Operation,
        prev_op: &Operation,
        is_valid: bool,
        range_checker: Arc<RangeCheckerGateChip>,
        oc_cols: &mut PageOfflineCheckerColsMut<F>,
    ) {
        let offline_checker_chip =
            OfflineCheckerChip::<F, Operation>::new(self.offline_checker.clone());

        offline_checker_chip.generate_trace_row(
            *is_first_row,
            is_valid,
            is_internal,
            curr_op,
            prev_op,
            range_checker.clone(),
            &mut oc_cols.offline_checker_cols,
        );

        if *is_first_row {
            *is_first_row = false;
            *oc_cols.offline_checker_cols.same_idx = F::zero();
        }

        let op_type = curr_op.op_type as u8;

        *oc_cols.is_initial = F::from_bool(is_initial);
        *oc_cols.is_final_write = F::from_bool(is_final && op_type == 0);
        *oc_cols.is_final_delete = F::from_bool(is_final && op_type == 2);
        *oc_cols.is_read = F::from_bool(op_type == 0);
        *oc_cols.is_write = F::from_bool(op_type == 1);
        *oc_cols.is_delete = F::from_bool(op_type == 2);
    }
}
