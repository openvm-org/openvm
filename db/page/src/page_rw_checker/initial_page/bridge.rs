use afs_stark_backend::interaction::InteractionBuilder;

use super::PageReadAir;
use crate::common::page_cols::PageCols;

impl PageReadAir {
    /// Sends page rows (idx, data) for every allocated row on page_bus
    /// Some of this is received by OfflineChecker and some by IndexedPageWriteAir
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        page: PageCols<AB::Var>,
    ) {
        let page_blob = page.idx.into_iter().chain(page.data);

        builder.push_send(self.page_bus, page_blob, page.is_alloc);
    }
}
