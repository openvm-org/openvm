use std::iter;

use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use super::columns::LeafPageCols;
use super::{LeafPageAir, PageRwAir};
use crate::indexed_output_page_air::columns::IndexedOutputPageCols;
use crate::is_less_than_tuple::columns::{IsLessThanTupleCols, IsLessThanTupleIOCols};
use crate::page_rw_checker::final_page::columns::IndexedPageWriteCols;
use crate::sub_chip::SubAirBridge;

impl<const COMMITMENT_LEN: usize> LeafPageAir<COMMITMENT_LEN> {
    fn custom_receives_path<F: PrimeField64>(
        &self,
        col_indices: LeafPageCols<usize>,
    ) -> Vec<Interaction<F>> {
        // Sending the path
        if self.is_init {
            let virtual_cols = (col_indices.metadata.own_commitment)
                .into_iter()
                .chain(iter::once(col_indices.metadata.air_id))
                .map(VirtualPairCol::single_main)
                .collect::<Vec<_>>();

            vec![Interaction {
                fields: virtual_cols,
                count: VirtualPairCol::single_main(col_indices.cache_cols.is_alloc),
                argument_index: *self.path_bus_index(),
            }]
        } else {
            let range_inclusion_cols = col_indices.metadata.range_inclusion_cols.unwrap();
            let virtual_cols = range_inclusion_cols
                .start
                .into_iter()
                .chain(range_inclusion_cols.end)
                .chain(col_indices.metadata.own_commitment)
                .chain(iter::once(col_indices.metadata.air_id))
                .map(VirtualPairCol::single_main)
                .collect::<Vec<_>>();

            vec![Interaction {
                fields: virtual_cols,
                count: VirtualPairCol::single_main(col_indices.cache_cols.is_alloc),
                argument_index: *self.path_bus_index(),
            }]
        }
    }
}

impl<F: PrimeField64, const COMMITMENT_LEN: usize> SubAirBridge<F> for LeafPageAir<COMMITMENT_LEN> {
    fn receives(&self, col_indices: LeafPageCols<usize>) -> Vec<Interaction<F>> {
        let mut interactions = vec![];
        match &self.page_chip {
            PageRwAir::Initial(i) => {
                interactions.extend(SubAirBridge::receives(i, col_indices.cache_cols.clone()));
            }
            PageRwAir::Final(fin) => {
                interactions.extend(SubAirBridge::receives(
                    fin,
                    IndexedPageWriteCols {
                        final_page_cols: IndexedOutputPageCols {
                            page_cols: col_indices.cache_cols.clone(),
                            aux_cols: col_indices
                                .metadata
                                .subair_aux_cols
                                .clone()
                                .unwrap()
                                .final_page_aux
                                .final_page_aux_cols,
                        },
                        rcv_mult: col_indices
                            .metadata
                            .subair_aux_cols
                            .clone()
                            .unwrap()
                            .final_page_aux
                            .rcv_mult,
                    },
                ));
            }
        };

        interactions.extend(self.custom_receives_path(col_indices.clone()));
        interactions
    }

    fn sends(&self, col_indices: LeafPageCols<usize>) -> Vec<Interaction<F>> {
        let mut interactions = vec![];
        match &self.page_chip {
            PageRwAir::Initial(i) => {
                interactions.extend(SubAirBridge::sends(i, col_indices.cache_cols.clone()));
            }
            PageRwAir::Final(fin) => {
                interactions.extend(SubAirBridge::sends(
                    fin,
                    IndexedPageWriteCols {
                        final_page_cols: IndexedOutputPageCols {
                            page_cols: col_indices.cache_cols.clone(),
                            aux_cols: col_indices
                                .metadata
                                .subair_aux_cols
                                .clone()
                                .unwrap()
                                .final_page_aux
                                .final_page_aux_cols,
                        },
                        rcv_mult: col_indices
                            .metadata
                            .subair_aux_cols
                            .clone()
                            .unwrap()
                            .final_page_aux
                            .rcv_mult,
                    },
                ));
            }
        };

        if !self.is_init {
            let subairs = self.is_less_than_tuple_air.clone().unwrap();
            let range_inclusion = col_indices.metadata.range_inclusion_cols.clone().unwrap();
            let subair_aux = col_indices.metadata.subair_aux_cols.clone().unwrap();
            interactions.extend(SubAirBridge::sends(
                &subairs.idx_start,
                IsLessThanTupleCols {
                    io: IsLessThanTupleIOCols {
                        x: col_indices.cache_cols.idx.clone(),
                        y: range_inclusion.start.clone(),
                        tuple_less_than: range_inclusion.less_than_start,
                    },
                    aux: subair_aux.idx_start.clone(),
                },
            ));
            interactions.extend(SubAirBridge::sends(
                &subairs.end_idx,
                IsLessThanTupleCols {
                    io: IsLessThanTupleIOCols {
                        x: range_inclusion.end.clone(),
                        y: col_indices.cache_cols.idx.clone(),
                        tuple_less_than: range_inclusion.greater_than_end,
                    },
                    aux: subair_aux.end_idx.clone(),
                },
            ));
        }
        interactions
    }
}

impl<F: PrimeField64, const COMMITMENT_LEN: usize> AirBridge<F> for LeafPageAir<COMMITMENT_LEN> {
    fn receives(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_to_receive = LeafPageCols::<usize>::from_slice(
            &all_cols,
            self.idx_len,
            self.data_len,
            COMMITMENT_LEN,
            self.is_init,
            self.is_less_than_tuple_param.clone(),
        );
        SubAirBridge::receives(self, cols_to_receive)
    }

    fn sends(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_to_receive = LeafPageCols::<usize>::from_slice(
            &all_cols,
            self.idx_len,
            self.data_len,
            COMMITMENT_LEN,
            self.is_init,
            self.is_less_than_tuple_param.clone(),
        );
        SubAirBridge::sends(self, cols_to_receive)
    }
}
