use afs_stark_backend::interaction::InteractionBuilder;
use itertools::Itertools;

use super::PageReadAir;
use crate::common::page_cols::PageCols;

impl PageReadAir {
    /// Sends page rows (idx, data) for every allocated row on page_bus
    /// Some of this is received by OfflineChecker and some by MyFinalPageChip
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &PageCols<AB::Var>,
    ) {
        let page_cols = cols
            .idx
            .clone()
            .into_iter()
            .chain(cols.data.clone())
            .collect_vec();

        builder.push_send(self.page_bus, page_cols, cols.is_alloc);
    }
}
