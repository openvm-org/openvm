use afs_stark_backend::interaction::InteractionBuilder;

use crate::common::page_cols::PageCols;

use super::PageReadAir;

impl PageReadAir {
    /// Sends page rows (idx, data) for every allocated row on page_bus
    /// Some of this is received by OfflineChecker and some by MyFinalPageChip
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        page: PageCols<AB::Var>,
    ) {
        let page_blob = page.idx.into_iter().chain(page.data);

        builder.push_send(self.page_bus, page_blob, page.is_alloc);
    }
}
