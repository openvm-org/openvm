use std::iter;

use afs_stark_backend::interaction::InteractionBuilder;

use super::{columns::RootSignalCols, RootSignalAir};

impl<const COMMITMENT_LEN: usize> RootSignalAir<COMMITMENT_LEN> {
    fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &RootSignalCols<AB::Var>,
    ) {
        if self.is_init {
            let virtual_cols = (cols.root_commitment.clone())
                .into_iter()
                .chain(iter::once(cols.air_id))
                .collect::<Vec<_>>();
            builder.push_send(*self.bus_index(), virtual_cols, cols.mult);
        } else {
            let virtual_cols = (cols.range.clone().unwrap().0)
                .into_iter()
                .chain(cols.range.clone().unwrap().1)
                .chain(cols.root_commitment.clone())
                .chain(iter::once(cols.air_id))
                .collect::<Vec<_>>();

            builder.push_send(*self.bus_index(), virtual_cols, cols.mult);
        }
    }
}
