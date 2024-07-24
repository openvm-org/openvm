use std::iter;

use afs_stark_backend::interaction::{InteractionBuilder, InteractionType};
use p3_air::AirBuilderWithPublicValues;
use p3_field::AbstractField;

use super::columns::InternalPageCols;
use super::InternalPageAir;

impl<const COMMITMENT_LEN: usize> InternalPageAir<COMMITMENT_LEN> {
    fn custom_receives_path<AB: InteractionBuilder + AirBuilderWithPublicValues>(
        &self,
        builder: &mut AB,
        page_cols: &InternalPageCols<impl Into<AB::Expr> + Clone>,
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
                page_cols.metadata.mult_alloc.clone(),
            );
        } else {
            let range_inclusion_cols = page_cols.metadata.range_inclusion_cols.as_ref().unwrap();
            let virtual_cols = range_inclusion_cols
                .start
                .iter()
                .map(|x| x.clone().into())
                .chain(range_inclusion_cols.end.iter().map(|x| x.clone().into()))
                .chain(own_commitment.iter().map(|x| (*x).into()))
                .chain(iter::once(AB::Expr::from_canonical_u32(self.air_id)))
                .collect::<Vec<_>>();

            builder.push_receive(
                *self.path_bus_index(),
                virtual_cols,
                page_cols.metadata.mult_alloc.clone(),
            );
        }
    }

    fn custom_sends_or_receives_data<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        page_cols: &InternalPageCols<impl Into<AB::Expr> + Clone>,
        is_send: bool,
    ) {
        let virtual_cols = iter::once(page_cols.cache_cols.is_alloc.clone())
            .chain(page_cols.cache_cols.child_start.clone())
            .chain(page_cols.cache_cols.child_end.clone())
            .chain(page_cols.cache_cols.commitment.clone())
            .collect::<Vec<_>>();
        builder.push_interaction(
            *self.data_bus_index(),
            virtual_cols,
            page_cols.metadata.mult_alloc_is_1.clone(),
            if is_send {
                InteractionType::Send
            } else {
                InteractionType::Receive
            },
        );
    }

    fn custom_sends_path<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        page_cols: &InternalPageCols<impl Into<AB::Expr> + Clone>,
    ) {
        // Sending the path
        if self.is_init {
            let virtual_cols = (page_cols.cache_cols.commitment.clone())
                .into_iter()
                .chain(iter::once(page_cols.metadata.child_air_id.clone()))
                .collect::<Vec<_>>();
            builder.push_send(
                *self.path_bus_index(),
                virtual_cols,
                page_cols.metadata.mult_minus_one_alloc.clone(),
            );
        } else {
            let virtual_cols = page_cols
                .cache_cols
                .child_start
                .clone()
                .into_iter()
                .chain(page_cols.cache_cols.child_end.clone())
                .chain(page_cols.cache_cols.commitment.clone())
                .chain(iter::once(page_cols.metadata.child_air_id.clone()))
                .collect::<Vec<_>>();

            builder.push_send(
                *self.path_bus_index(),
                virtual_cols,
                page_cols.metadata.mult_minus_one_alloc.clone(),
            );
        }
    }
}

impl<const COMMITMENT_LEN: usize> InternalPageAir<COMMITMENT_LEN> {
    pub fn eval_interactions<AB: InteractionBuilder + AirBuilderWithPublicValues>(
        &self,
        builder: &mut AB,
        page_cols: &InternalPageCols<AB::Var>,
        own_commitment: &[AB::PublicVar],
    ) {
        self.custom_receives_path(builder, page_cols, own_commitment);
        self.custom_sends_or_receives_data(builder, page_cols, self.is_init);
        self.custom_sends_path(builder, page_cols);
        if !self.is_init {
            let subairs = self.is_less_than_tuple_air.clone().unwrap();
            let subair_aux = page_cols.metadata.subair_aux_cols.clone().unwrap();
            subairs
                .idx1_start
                .eval_interactions(builder, &subair_aux.idx1_start.less_than_aux);
            subairs
                .end_idx2
                .eval_interactions(builder, &subair_aux.end_idx2.less_than_aux);
            subairs
                .idx2_next
                .eval_interactions(builder, &subair_aux.idx2_next.less_than_aux);
            subairs
                .idx2_idx1
                .eval_interactions(builder, &subair_aux.idx2_idx1.less_than_aux);
        }
    }
}

// impl<F: PrimeField64, const COMMITMENT_LEN: usize> AirBridge<F>
//     for InternalPageAir<COMMITMENT_LEN>
// {
//     fn receives(&self) -> Vec<Interaction<F>> {
//         let num_cols = self.air_width();
//         let all_cols = (0..num_cols).collect::<Vec<usize>>();

//         let cols_to_receive = InternalPageCols::<usize>::from_slice(
//             &all_cols,
//             self.idx_len,
//             COMMITMENT_LEN,
//             self.is_init,
//             self.is_less_than_tuple_param.clone(),
//         );
//         SubAirBridge::receives(self, cols_to_receive)
//     }

//     fn sends(&self) -> Vec<Interaction<F>> {
//         let num_cols = self.air_width();
//         let all_cols = (0..num_cols).collect::<Vec<usize>>();

//         let cols_to_receive = InternalPageCols::<usize>::from_slice(
//             &all_cols,
//             self.idx_len,
//             COMMITMENT_LEN,
//             self.is_init,
//             self.is_less_than_tuple_param.clone(),
//         );
//         SubAirBridge::sends(self, cols_to_receive)
//     }
// }
