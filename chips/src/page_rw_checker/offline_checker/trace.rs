use std::sync::Arc;

use p3_field::{AbstractField, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};

use super::OfflineChecker;
use crate::common::indexed_page_editor::IndexedPageEditor;
use crate::common::page::Page;
use crate::offline_checker::GeneralOfflineCheckerChip;
use crate::page_rw_checker::offline_checker::columns::OfflineCheckerCols;
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
        Val<SC>: PrimeField64,
    {
        let mut page_editor = IndexedPageEditor::from_page(page);

        // Creating a timestamp bigger than all others
        let max_clk = ops.iter().map(|op| op.clk).max().unwrap_or(0) + 1;

        ops.sort_by_key(|op| (op.idx.clone(), op.clk));

        let dummy_op = Operation {
            idx: vec![0; self.general_offline_checker.idx_len],
            data: vec![0; self.general_offline_checker.data_len],
            op_type: OpType::Read,
            clk: 0,
        };

        // This takes the information for the current row and references for the last row
        // It uses those values to generate the new row in the trace, and it updates the references
        // to the new row's information
        let gen_row = |is_first_row: &mut bool,
                       is_initial: bool,
                       is_final: bool,
                       is_internal: bool,
                       curr_op: Operation,
                       prev_op: Operation,
                       is_valid: bool| {
            let local_input = (
                *is_first_row,
                is_valid as u8,
                curr_op.clone(),
                prev_op.clone(),
                range_checker.clone(),
            );

            let general_oc_chip = GeneralOfflineCheckerChip::<Val<SC>, Operation>::new(
                self.general_offline_checker.clone(),
            );

            let mut general_oc_cols = LocalTraceInstructions::<Val<SC>>::generate_trace_row(
                &general_oc_chip,
                local_input,
            );

            if *is_first_row {
                *is_first_row = false;
                general_oc_cols.same_idx = Val::<SC>::zero();
                general_oc_cols.same_data = Val::<SC>::zero();
            }

            let op_type = curr_op.op_type as u8;

            let is_final_write = is_final && op_type == 0;
            let is_final_delete = is_final && op_type == 2;
            let is_read = op_type == 0;
            let is_write = op_type == 1;
            let is_delete = op_type == 2;

            let cols = OfflineCheckerCols {
                general_cols: general_oc_cols,
                is_initial: Val::<SC>::from_bool(is_initial),
                is_final_write: Val::<SC>::from_bool(is_final_write),
                is_final_delete: Val::<SC>::from_bool(is_final_delete),
                is_internal: Val::<SC>::from_bool(is_internal),
                is_final_write_x3: Val::<SC>::from_canonical_u8(is_final_write as u8 * 3),
                is_read: Val::<SC>::from_bool(is_read),
                is_write: Val::<SC>::from_bool(is_write),
                is_delete: Val::<SC>::from_bool(is_delete),
            };

            assert!(cols.flatten().len() == self.air_width());
            cols.flatten()
        };

        let mut rows = vec![];

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
                prev_op = curr_op.clone();
                curr_op = Operation {
                    idx: cur_idx.clone(),
                    data: page_editor.get(&cur_idx).unwrap().clone(),
                    op_type: OpType::Write,
                    clk: 0,
                };

                // Adding the is_initial row to the trace
                rows.push(gen_row(
                    &mut is_first_row,
                    true,
                    false,
                    false,
                    curr_op.clone(),
                    prev_op.clone(),
                    true,
                ));
            }

            for op in ops.iter().take(j).skip(i) {
                prev_op = curr_op.clone();
                curr_op = op.clone();

                if op.op_type == OpType::Write {
                    page_editor.write(&cur_idx, &op.data);
                } else if op.op_type == OpType::Delete {
                    page_editor.delete(&cur_idx);
                }

                rows.push(gen_row(
                    &mut is_first_row,
                    false,
                    false,
                    true,
                    curr_op.clone(),
                    prev_op.clone(),
                    true,
                ));
            }

            let final_data = page_editor
                .get(&cur_idx)
                .cloned()
                .unwrap_or(vec![0; self.general_offline_checker.data_len]);

            prev_op = curr_op.clone();
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
            rows.push(gen_row(
                &mut is_first_row,
                false,
                true,
                false,
                curr_op.clone(),
                prev_op.clone(),
                true,
            ));

            i = j;
        }

        // Ensure that trace degree is a power of two
        assert!(trace_degree > 0 && trace_degree & (trace_degree - 1) == 0);

        *page = page_editor.into_page();

        if rows.len() < trace_degree {
            prev_op = curr_op.clone();
            curr_op = dummy_op.clone();

            rows.push(gen_row(
                &mut is_first_row,
                false,
                false,
                false,
                curr_op.clone(),
                prev_op.clone(),
                false,
            ));
        }

        prev_op = dummy_op.clone();

        // Adding rows to the trace to make the height trace_degree
        rows.resize_with(trace_degree, || {
            gen_row(
                &mut is_first_row,
                false,
                false,
                false,
                curr_op.clone(),
                prev_op.clone(),
                false,
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
