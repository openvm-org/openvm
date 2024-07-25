use afs_stark_backend::interaction::InteractionBuilder;

use super::ReceivingIndexedOutputPageAir;
use crate::common::page_cols::PageCols;

// Receiving page columns from groupby air
impl ReceivingIndexedOutputPageAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        page_cols: PageCols<AB::Var>,
    ) {
        // let page_cols = local.final_page_cols.page_cols;
        let alloc_idx = page_cols.is_alloc;

        let page_cols = page_cols.to_vec().into_iter().skip(1).collect::<Vec<_>>();

        builder.push_receive(self.page_bus_index, page_cols, alloc_idx);
    }
}
