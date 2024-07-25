use std::iter;

use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::AirBuilderWithPublicValues;
use p3_field::AbstractField;

use super::columns::LeafPageCols;
use super::LeafPageAir;

impl<const COMMITMENT_LEN: usize> LeafPageAir<COMMITMENT_LEN> {
    fn custom_receives_path<AB: InteractionBuilder + AirBuilderWithPublicValues>(
        &self,
        builder: &mut AB,
        page_cols: &LeafPageCols<AB::Var>,
        own_commitment: &[AB::PublicVar],
    ) {
        // Sending the path
        if self.is_init {
            let virtual_cols = own_commitment
                .iter()
                .map(|x| (*x).into())
                .chain(iter::once(AB::Expr::from_canonical_u32(self.air_id)))
                .collect::<Vec<_>>();
            builder.push_receive(
                *self.path_bus_index(),
                virtual_cols,
                page_cols.cache_cols.is_alloc,
            );
        } else {
            let range_inclusion_cols = page_cols.metadata.range_inclusion_cols.as_ref().unwrap();
            let virtual_cols = range_inclusion_cols
                .start
                .iter()
                .map(|x| (*x).into())
                .chain(range_inclusion_cols.end.iter().map(|x| (*x).into()))
                .chain(own_commitment.iter().map(|x| (*x).into()))
                .chain(iter::once(AB::Expr::from_canonical_u32(self.air_id)))
                .collect::<Vec<_>>();

            builder.push_receive(
                *self.path_bus_index(),
                virtual_cols,
                page_cols.cache_cols.is_alloc,
            );
        }
    }
}

impl<const COMMITMENT_LEN: usize> LeafPageAir<COMMITMENT_LEN> {
    pub fn eval_interactions<AB: InteractionBuilder + AirBuilderWithPublicValues>(
        &self,
        builder: &mut AB,
        page_cols: &LeafPageCols<AB::Var>,
        own_commitment: &[AB::PublicVar],
    ) {
        self.custom_receives_path(builder, page_cols, own_commitment);
    }
}
