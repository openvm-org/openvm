use ax_stark_backend::interaction::InteractionBuilder;
use itertools::Itertools;

use super::{columns::OfflineCheckerCols, OfflineChecker};

impl OfflineChecker {
    /// Receives operations (clk, op_type, idx, data) for rows tagged with is_internal on ops_bus
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &OfflineCheckerCols<AB::Var>,
    ) {
        let op_cols = [cols.clk, cols.op_type]
            .into_iter()
            .chain(cols.idx.clone())
            .chain(cols.data.clone())
            .collect_vec();
        builder.push_receive(self.ops_bus, op_cols, cols.is_receive);
    }
}
