use std::iter;

use afs_stark_backend::interaction::InteractionBuilder;
use itertools::Itertools;

use super::columns::OfflineCheckerCols;
use super::OfflineChecker;

impl OfflineChecker {
    /// Receives page rows (idx, data) for rows tagged with is_initial on page_bus (sent from PageRWAir)
    /// Receives operations (clk, idx, data, op_type) for rows tagged with is_internal on ops_bus
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &OfflineCheckerCols<AB::Var>,
    ) {
        let page_cols = cols.idx.iter().chain(&cols.data).cloned().collect_vec();

        let op_cols = iter::once(cols.clk)
            .chain(cols.idx.clone())
            .chain(cols.data.clone())
            .chain(iter::once(cols.op_type))
            .collect_vec();
        builder.push_receive(self.page_bus_index, page_cols.clone(), cols.is_initial);
        builder.push_receive(self.ops_bus_index, op_cols, cols.is_internal);
        builder.push_send(self.page_bus_index, page_cols, cols.is_final_write_x3);
    }
}
