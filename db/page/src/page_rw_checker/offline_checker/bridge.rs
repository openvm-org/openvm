use afs_stark_backend::interaction::InteractionBuilder;
use itertools::Itertools;
use p3_field::AbstractField;

use super::{columns::PageOfflineCheckerCols, PageOfflineChecker};

impl PageOfflineChecker {
    /// Receives page rows (idx, data) for rows tagged with is_initial on page_bus (sent from PageReadAir)
    /// Sends page rows (idx, data) for rows tagged with is_final_write on page_bus with multiplicity is_final_write * 3 (received by IndexedPageWriteAir)
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &PageOfflineCheckerCols<AB::Var>,
    ) {
        let idx = &cols.offline_checker_cols.idx;
        let data = &cols.offline_checker_cols.data;
        let page_cols = idx.iter().chain(data).cloned().collect_vec();

        builder.push_receive(self.page_bus_index, page_cols.clone(), cols.is_initial);
        builder.push_send(
            self.page_bus_index,
            page_cols,
            cols.is_final_write * AB::Expr::from_canonical_u32(3),
        );
    }
}
