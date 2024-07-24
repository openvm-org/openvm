use afs_stark_backend::interaction::InteractionBuilder;
use itertools::Itertools;

use super::FinalTableAir;
use crate::common::page_cols::PageCols;

impl FinalTableAir {
    /// Receives (idx, data) of T1 for every allocated row on t1_output_bus (sent by t1_chip)
    /// Receives (idx, data) of T2 for every allocated row on t2_output_bus (sent by t2_chip)
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        mut page: PageCols<AB::Var>,
    ) {
        let t1_data = page.data.split_off(self.t2_data_len);
        let t1_idx_data = page.data[self.fkey_start..self.fkey_end]
            .iter()
            .copied()
            .chain(t1_data)
            .collect_vec();

        let t2_idx_data = page.idx.into_iter().chain(page.data);

        builder.push_receive(self.buses.t1_output_bus_index, t1_idx_data, page.is_alloc);
        builder.push_receive(self.buses.t2_output_bus_index, t2_idx_data, page.is_alloc);
    }
}
