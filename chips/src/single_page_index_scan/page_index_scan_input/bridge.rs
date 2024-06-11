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
        let mut interactions: Vec<Interaction<F>> = vec![];

        match &self {
            PageIndexScanInputAir::Lt {
                bus_index,
                idx_len,
                data_len,
                is_less_than_tuple_air,
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

                match cols_numbered {
                    PageIndexScanInputCols::Lt {
                        is_alloc,
                        idx,
                        data,
                        x,
                        satisfies_pred,
                        send_row,
                        is_less_than_tuple_aux,
                        ..
                    } => {
                        let is_less_than_tuple_cols = IsLessThanTupleCols {
                            io: IsLessThanTupleIOCols {
                                x: idx.clone(),
                                y: x.clone(),
                                tuple_less_than: satisfies_pred,
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
                    PageIndexScanInputCols::Lte { .. } => {
                        panic!(
                            "expected PageIndexScanInputCols::Lt, got PageIndexScanInputCols::Lte"
                        );
                    }
                    PageIndexScanInputCols::Eq { .. } => {
                        panic!(
                            "expected PageIndexScanInputCols::Lt, got PageIndexScanInputCols::Eq"
                        );
                    }
                    PageIndexScanInputCols::Gte { .. } => {
                        panic!(
                            "expected PageIndexScanInputCols::Lt, got PageIndexScanInputCols::Gte"
                        );
                    }
                    PageIndexScanInputCols::Gt { .. } => {
                        panic!(
                            "expected PageIndexScanInputCols::Lt, got PageIndexScanInputCols::Gt"
                        );
                    }
                }

                interactions
            }
            PageIndexScanInputAir::Lte {
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
                    Comp::Lte,
                );

                let all_cols = (0..num_cols).collect::<Vec<usize>>();

                let cols_numbered = PageIndexScanInputCols::<usize>::from_slice(
                    &all_cols,
                    *idx_len,
                    *data_len,
                    is_less_than_tuple_air.limb_bits(),
                    is_less_than_tuple_air.decomp(),
                    Comp::Lte,
                );

                match cols_numbered {
                    PageIndexScanInputCols::Lt { .. } => {
                        panic!(
                            "expected PageIndexScanInputCols::Lte, got PageIndexScanInputCols::Lt"
                        );
                    }
                    PageIndexScanInputCols::Lte {
                        is_alloc,
                        idx,
                        data,
                        x,
                        satisfies_pred,
                        send_row,
                        is_less_than_tuple_aux,
                        ..
                    } => {
                        let is_less_than_tuple_cols = IsLessThanTupleCols {
                            io: IsLessThanTupleIOCols {
                                x: idx.clone(),
                                y: x.clone(),
                                tuple_less_than: satisfies_pred,
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
                    PageIndexScanInputCols::Eq { .. } => {
                        panic!(
                            "expected PageIndexScanInputCols::Lte, got PageIndexScanInputCols::Eq"
                        );
                    }
                    PageIndexScanInputCols::Gte { .. } => {
                        panic!(
                            "expected PageIndexScanInputCols::Lte, got PageIndexScanInputCols::Gte"
                        );
                    }
                    PageIndexScanInputCols::Gt { .. } => {
                        panic!(
                            "expected PageIndexScanInputCols::Lte, got PageIndexScanInputCols::Gt"
                        );
                    }
                }

                interactions
            }
            PageIndexScanInputAir::Eq {
                bus_index,
                idx_len,
                data_len,
                ..
            } => {
                // There is no limb_bits or decomp for IsEqualVec, so we can just pass in an empty vec and 0, respectively
                let num_cols = PageIndexScanInputCols::<F>::get_width(
                    *idx_len,
                    *data_len,
                    vec![],
                    0,
                    Comp::Eq,
                );

                let all_cols = (0..num_cols).collect::<Vec<usize>>();

                let cols_numbered = PageIndexScanInputCols::<usize>::from_slice(
                    &all_cols,
                    *idx_len,
                    *data_len,
                    vec![],
                    0,
                    Comp::Eq,
                );

                match cols_numbered {
                    PageIndexScanInputCols::Lt { .. } => {
                        panic!(
                            "expected PageIndexScanInputCols::Eq, got PageIndexScanInputCols::Lt"
                        );
                    }
                    PageIndexScanInputCols::Lte { .. } => {
                        panic!(
                            "expected PageIndexScanInputCols::Eq, got PageIndexScanInputCols::Lte"
                        );
                    }
                    PageIndexScanInputCols::Eq {
                        is_alloc,
                        idx,
                        data,
                        send_row,
                        ..
                    } => {
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
                    }
                    PageIndexScanInputCols::Gte { .. } => {
                        panic!(
                            "expected PageIndexScanInputCols::Eq, got PageIndexScanInputCols::Gte"
                        );
                    }
                    PageIndexScanInputCols::Gt { .. } => {
                        panic!(
                            "expected PageIndexScanInputCols::Eq, got PageIndexScanInputCols::Gt"
                        );
                    }
                }

                interactions
            }
            PageIndexScanInputAir::Gte {
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
                    Comp::Gte,
                );

                let all_cols = (0..num_cols).collect::<Vec<usize>>();

                let cols_numbered = PageIndexScanInputCols::<usize>::from_slice(
                    &all_cols,
                    *idx_len,
                    *data_len,
                    is_less_than_tuple_air.limb_bits(),
                    is_less_than_tuple_air.decomp(),
                    Comp::Gte,
                );

                match cols_numbered {
                    PageIndexScanInputCols::Lt { .. } => {
                        panic!(
                            "expected PageIndexScanInputCols::Gte, got PageIndexScanInputCols::Lt"
                        );
                    }
                    PageIndexScanInputCols::Lte { .. } => {
                        panic!(
                            "expected PageIndexScanInputCols::Gte, got PageIndexScanInputCols::Lte"
                        );
                    }
                    PageIndexScanInputCols::Eq { .. } => {
                        panic!(
                            "expected PageIndexScanInputCols::Gte, got PageIndexScanInputCols::Eq"
                        );
                    }
                    PageIndexScanInputCols::Gte {
                        is_alloc,
                        idx,
                        data,
                        x,
                        satisfies_pred,
                        send_row,
                        is_less_than_tuple_aux,
                        ..
                    } => {
                        let is_less_than_tuple_cols = IsLessThanTupleCols {
                            io: IsLessThanTupleIOCols {
                                x: x.clone(),
                                y: idx.clone(),
                                tuple_less_than: satisfies_pred,
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
                            "expected PageIndexScanInputCols::Gte, got PageIndexScanInputCols::Gt"
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

                match cols_numbered {
                    PageIndexScanInputCols::Lt { .. } => {
                        panic!(
                            "expected PageIndexScanInputCols::Gt, got PageIndexScanInputCols::Lt"
                        );
                    }
                    PageIndexScanInputCols::Lte { .. } => {
                        panic!(
                            "expected PageIndexScanInputCols::Gt, got PageIndexScanInputCols::Lte"
                        );
                    }
                    PageIndexScanInputCols::Eq { .. } => {
                        panic!(
                            "expected PageIndexScanInputCols::Gt, got PageIndexScanInputCols::Eq"
                        );
                    }
                    PageIndexScanInputCols::Gte { .. } => {
                        panic!(
                            "expected PageIndexScanInputCols::Gt, got PageIndexScanInputCols::Gte"
                        );
                    }
                    PageIndexScanInputCols::Gt {
                        is_alloc,
                        idx,
                        data,
                        x,
                        send_row,
                        satisfies_pred,
                        is_less_than_tuple_aux,
                        ..
                    } => {
                        let is_less_than_tuple_cols = IsLessThanTupleCols {
                            io: IsLessThanTupleIOCols {
                                x: x.clone(),
                                y: idx.clone(),
                                tuple_less_than: satisfies_pred,
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
