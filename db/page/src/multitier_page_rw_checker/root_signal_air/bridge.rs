use std::iter;

use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::AirBuilderWithPublicValues;

use super::{columns::RootSignalCols, RootSignalAir};

impl<const COMMITMENT_LEN: usize> RootSignalAir<COMMITMENT_LEN> {
    pub fn eval_interactions<AB: InteractionBuilder + AirBuilderWithPublicValues>(
        &self,
        builder: &mut AB,
        cols: &RootSignalCols<AB::Var>,
        own_commitment: &[AB::PublicVar],
    ) {
        if self.is_init {
            let virtual_cols = own_commitment
                .iter()
                .map(|x| (*x).into())
                .chain(iter::once(cols.air_id.into()))
                .collect::<Vec<_>>();
            builder.push_send(*self.bus_index(), virtual_cols, cols.mult);
        } else {
            let virtual_cols = (cols.range.clone().unwrap().0.iter().map(|x| (*x).into()))
                .chain(cols.range.clone().unwrap().1.iter().map(|x| (*x).into()))
                .chain(own_commitment.iter().map(|x| (*x).into()))
                .chain(iter::once(cols.air_id.into()))
                .collect::<Vec<_>>();

            builder.push_send(*self.bus_index(), virtual_cols, cols.mult);
        }
    }
}
