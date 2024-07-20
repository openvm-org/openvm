use afs_stark_backend::interaction::InteractionBuilder;
use itertools::Itertools;

use super::{columns::IndexedPageWriteCols, IndexedPageWriteAir};

impl IndexedPageWriteAir {
    /// Sends interactions required by IsLessThanTuple SubAir
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &IndexedPageWriteCols<AB::Var>,
    ) {
        let page_cols = cols.final_page_cols.page_cols.clone();
        let rcv_mult = cols.rcv_mult;
        let page_cols = page_cols
            .idx
            .clone()
            .into_iter()
            .chain(page_cols.data.clone())
            .collect_vec();
        builder.push_receive(self.page_bus_index, page_cols, rcv_mult);
    }
}
