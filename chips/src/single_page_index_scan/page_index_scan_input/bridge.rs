use crate::{
    is_less_than_tuple::columns::{IsLessThanTupleCols, IsLessThanTupleIOCols},
    sub_chip::SubAirBridge,
};

use super::{columns::PageIndexScanInputCols, Comp};
use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use super::PageIndexScanInputAir;

impl<F: PrimeField64> AirBridge<F> for PageIndexScanInputAir {
    fn sends(&self) -> Vec<Interaction<F>> {
        match &self {
            PageIndexScanInputAir::Lt {
                bus_index,
                idx_len,
                data_len,
                is_less_than_tuple_air,
                ..
            } => {
                let num_cols = PageIndexScanInputCols::<F>::get_width(
                    *idx_len,
                    *data_len,
                    is_less_than_tuple_air.limb_bits(),
                    is_less_than_tuple_air.decomp(),
                    Comp::Lt,
                );
                let all_cols = (0..num_cols).collect::<Vec<usize>>();

                let cols_numbered = PageIndexScanInputCols::<usize>::from_slice(
                    &all_cols,
                    *idx_len,
                    *data_len,
                    is_less_than_tuple_air.limb_bits(),
                    is_less_than_tuple_air.decomp(),
                    Comp::Lt,
                );

                let mut interactions: Vec<Interaction<F>> = vec![];

                match cols_numbered {
                    PageIndexScanInputCols::Lt {
                        is_alloc,
                        idx,
                        data,
                        send_row,
                        is_less_than_tuple_aux,
                        ..
                    } => {
                        let is_less_than_tuple_cols = IsLessThanTupleCols {
                            io: IsLessThanTupleIOCols {
                                x: idx.clone(),
                                y: data.clone(),
                                tuple_less_than: send_row,
                            },
                            aux: is_less_than_tuple_aux,
                        };

                        // construct the row to send
                        let mut cols = vec![];
                        cols.push(is_alloc);
                        cols.extend(idx);
                        cols.extend(data);

                        let virtual_cols = cols
                            .iter()
                            .map(|col| VirtualPairCol::single_main(*col))
                            .collect::<Vec<_>>();

                        interactions.push(Interaction {
                            fields: virtual_cols,
                            count: VirtualPairCol::single_main(send_row),
                            argument_index: *bus_index,
                        });

                        let mut subchip_interactions = SubAirBridge::<F>::sends(
                            is_less_than_tuple_air,
                            is_less_than_tuple_cols,
                        );

                        interactions.append(&mut subchip_interactions);
                    }
                    PageIndexScanInputCols::Gt { .. } => {
                        panic!(
                            "expected PageIndexScanInputCols::Lt, got PageIndexScanInputCols::Gt"
                        );
                    }
                }

                interactions
            }

            PageIndexScanInputAir::Gt {
                bus_index,
                idx_len,
                data_len,
                is_less_than_tuple_air,
                ..
            } => {
                let num_cols = PageIndexScanInputCols::<F>::get_width(
                    *idx_len,
                    *data_len,
                    is_less_than_tuple_air.limb_bits(),
                    is_less_than_tuple_air.decomp(),
                    Comp::Gt,
                );
                let all_cols = (0..num_cols).collect::<Vec<usize>>();

                let cols_numbered = PageIndexScanInputCols::<usize>::from_slice(
                    &all_cols,
                    *idx_len,
                    *data_len,
                    is_less_than_tuple_air.limb_bits(),
                    is_less_than_tuple_air.decomp(),
                    Comp::Gt,
                );

                let mut interactions: Vec<Interaction<F>> = vec![];

                match cols_numbered {
                    PageIndexScanInputCols::Lt { .. } => {
                        panic!(
                            "expected PageIndexScanInputCols::Lt, got PageIndexScanInputCols::Gt"
                        );
                    }
                    PageIndexScanInputCols::Gt {
                        is_alloc,
                        idx,
                        data,
                        send_row,
                        is_less_than_tuple_aux,
                        ..
                    } => {
                        let is_less_than_tuple_cols = IsLessThanTupleCols {
                            io: IsLessThanTupleIOCols {
                                x: idx.clone(),
                                y: data.clone(),
                                tuple_less_than: send_row,
                            },
                            aux: is_less_than_tuple_aux,
                        };

                        // construct the row to send
                        let mut cols = vec![];
                        cols.push(is_alloc);
                        cols.extend(idx);
                        cols.extend(data);

                        let virtual_cols = cols
                            .iter()
                            .map(|col| VirtualPairCol::single_main(*col))
                            .collect::<Vec<_>>();

                        interactions.push(Interaction {
                            fields: virtual_cols,
                            count: VirtualPairCol::single_main(send_row),
                            argument_index: *bus_index,
                        });

                        let mut subchip_interactions = SubAirBridge::<F>::sends(
                            is_less_than_tuple_air,
                            is_less_than_tuple_cols,
                        );

                        interactions.append(&mut subchip_interactions);
                    }
                }

                interactions
            }
        }
    }
}
