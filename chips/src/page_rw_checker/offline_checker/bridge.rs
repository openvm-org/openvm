use afs_stark_backend::interaction::InteractionBuilder;
use itertools::Itertools;

use super::{columns::PageOfflineCheckerCols, PageOfflineChecker};

impl PageOfflineChecker {
    /// Receives page rows (idx, data) for rows tagged with is_initial on page_bus (sent from PageRWAir)
    /// Receives operations (clk, idx, data, op_type) for rows tagged with is_internal on ops_bus
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &PageOfflineCheckerCols<AB::Var>,
    ) {
        let idx = &cols.offline_checker_cols.idx;
        let data = &cols.offline_checker_cols.data;
        let page_cols = idx.iter().chain(data).cloned().collect_vec();
        let op_cols = [
            cols.offline_checker_cols.clk,
            cols.offline_checker_cols.op_type,
        ]
        .iter()
        .chain(idx)
        .chain(data)
        .cloned()
        .collect_vec();

        builder.push_send(
            self.offline_checker.ops_bus,
            op_cols.clone(),
            cols.is_initial,
        );
        builder.push_send(
            self.offline_checker.ops_bus,
            op_cols.clone(),
            cols.is_final_delete,
        );
        builder.push_send(
            self.offline_checker.ops_bus,
            op_cols.clone(),
            cols.is_final_write,
        );
        builder.push_receive(self.page_bus_index, page_cols.clone(), cols.is_initial);
        builder.push_send(self.page_bus_index, page_cols, cols.is_final_write_x3);
    }
}
