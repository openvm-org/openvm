use std::iter;

use afs_stark_backend::interaction::{Interaction, InteractionBuilder};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use super::columns::LeafPageCols;
use super::{LeafPageAir, PageRwAir};
use crate::indexed_output_page_air::columns::IndexedOutputPageCols;
use crate::is_less_than_tuple::columns::{IsLessThanTupleCols, IsLessThanTupleIoCols};
use crate::page_rw_checker::final_page::columns::IndexedPageWriteCols;

impl<const COMMITMENT_LEN: usize> LeafPageAir<COMMITMENT_LEN> {
    fn custom_receives_path<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        page_cols: LeafPageCols<AB::Var>,
    ) {
        // Sending the path
        if self.is_init {
            let virtual_cols = (page_cols.metadata.own_commitment)
                .into_iter()
                .chain(iter::once(page_cols.metadata.air_id))
                .collect::<Vec<_>>();
            builder.push_receive(
                *self.path_bus_index(),
                virtual_cols,
                page_cols.cache_cols.is_alloc,
            );
        } else {
            let range_inclusion_cols = page_cols.metadata.range_inclusion_cols.unwrap();
            let virtual_cols = range_inclusion_cols
                .start
                .into_iter()
                .chain(range_inclusion_cols.end)
                .chain(page_cols.metadata.own_commitment)
                .chain(iter::once(page_cols.metadata.air_id))
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
    fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        page_cols: &LeafPageCols<AB::Var>,
    ) {
        let mut interactions = vec![];
        match &self.page_chip {
            PageRwAir::Initial(i) => {
                i.eval_interactions(builder, page_cols.cache_cols.clone());
            }
            PageRwAir::Final(fin) => {
                fin.eval_interactions(
                    builder,
                    IndexedPageWriteCols {
                        final_page_cols: IndexedOutputPageCols {
                            page_cols: page_cols.cache_cols.clone(),
                            aux_cols: page_cols
                                .metadata
                                .subair_aux_cols
                                .clone()
                                .unwrap()
                                .final_page_aux
                                .final_page_aux_cols
                                .clone(),
                        },
                        rcv_mult: page_cols
                            .metadata
                            .subair_aux_cols
                            .clone()
                            .unwrap()
                            .final_page_aux
                            .rcv_mult,
                    },
                );
            }
        };

        self.custom_receives_path(builder, page_cols);
        if !self.is_init {
            let subairs = self.is_less_than_tuple_air.clone().unwrap();
            let range_inclusion = page_cols.metadata.range_inclusion_cols.clone().unwrap();
            let subair_aux = page_cols.metadata.subair_aux_cols.clone().unwrap();
            subairs
                .idx_start
                .eval_interactions(builder, subair_aux.idx_start.clone());
            subairs
                .end_idx
                .eval_interactions(builder, subair_aux.end_idx.clone());
        }
    }
}
