use afs_stark_backend::air_builders::PartitionedAirBuilder;
use p3_air::{Air, AirBuilderWithPublicValues, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use crate::{
    is_equal_vec::columns::{IsEqualVecCols, IsEqualVecIOCols},
    is_less_than_tuple::columns::{IsLessThanTupleCols, IsLessThanTupleIOCols},
    sub_chip::{AirConfig, SubAir},
};

use super::{columns::PageIndexScanInputCols, Comp, PageIndexScanInputAir};

impl AirConfig for PageIndexScanInputAir {
    type Cols<T> = PageIndexScanInputCols<T>;
}

impl<F: Field> BaseAir<F> for PageIndexScanInputAir {
    fn width(&self) -> usize {
        match &self {
            PageIndexScanInputAir::Lt {
                idx_len,
                data_len,
                is_less_than_tuple_air,
                ..
            }
            | PageIndexScanInputAir::Gt {
                idx_len,
                data_len,
                is_less_than_tuple_air,
                ..
            } => PageIndexScanInputCols::<F>::get_width(
                *idx_len,
                *data_len,
                is_less_than_tuple_air.limb_bits().clone(),
                is_less_than_tuple_air.decomp(),
                Comp::Lt,
            ),
            PageIndexScanInputAir::Lte {
                idx_len,
                data_len,
                is_less_than_tuple_air,
                ..
            }
            | PageIndexScanInputAir::Gte {
                idx_len,
                data_len,
                is_less_than_tuple_air,
                ..
            } => PageIndexScanInputCols::<F>::get_width(
                *idx_len,
                *data_len,
                is_less_than_tuple_air.limb_bits().clone(),
                is_less_than_tuple_air.decomp(),
                Comp::Lte,
            ),
            // there is no idx_limb_bits or decomp, so we supply an empty vec and 0, respectively
            PageIndexScanInputAir::Eq {
                idx_len, data_len, ..
            } => PageIndexScanInputCols::<F>::get_width(*idx_len, *data_len, vec![], 0, Comp::Eq),
        }
    }
}

impl<AB: PartitionedAirBuilder + AirBuilderWithPublicValues> Air<AB> for PageIndexScanInputAir
where
    AB::M: Clone,
{
    fn eval(&self, builder: &mut AB) {
        match &self {
            PageIndexScanInputAir::Lt {
                idx_len,
                data_len,
                is_less_than_tuple_air,
                ..
            } => {
                let page_main = &builder.partitioned_main()[0].clone();
                let aux_main = &builder.partitioned_main()[1].clone();

                // get the public value x
                let pis = builder.public_values();
                let public_x = pis[..*idx_len].to_vec();

                let local_page = page_main.row_slice(0);
                let local_aux = aux_main.row_slice(0);
                let local_vec = local_page
                    .iter()
                    .chain(local_aux.iter())
                    .cloned()
                    .collect::<Vec<AB::Var>>();
                let local = local_vec.as_slice();

                let local_cols = PageIndexScanInputCols::<AB::Var>::from_slice(
                    local,
                    *idx_len,
                    *data_len,
                    is_less_than_tuple_air.limb_bits().clone(),
                    is_less_than_tuple_air.decomp(),
                    Comp::Lt,
                );

                match local_cols {
                    PageIndexScanInputCols::Lt {
                        is_alloc,
                        idx,
                        x,
                        satisfies_pred,
                        send_row,
                        is_less_than_tuple_aux,
                        ..
                    } => {
                        // here, we are checking if idx < x
                        let is_less_than_tuple_cols = IsLessThanTupleCols {
                            io: IsLessThanTupleIOCols {
                                x: idx,
                                y: x.clone(),
                                tuple_less_than: satisfies_pred,
                            },
                            aux: is_less_than_tuple_aux,
                        };

                        // constrain that the public value x is the same as the column x
                        for (&local_x, &pub_x) in x.iter().zip(public_x.iter()) {
                            builder.assert_eq(local_x, pub_x);
                        }

                        // constrain that we send the row iff the row is allocated and satisfies the predicate
                        builder.assert_eq(is_alloc * satisfies_pred, send_row);
                        builder.assert_bool(send_row);

                        // constrain the indicator that we used to check wheter key < x is correct
                        SubAir::eval(
                            is_less_than_tuple_air,
                            &mut builder.when_transition(),
                            is_less_than_tuple_cols.io,
                            is_less_than_tuple_cols.aux,
                        );
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
            }
            PageIndexScanInputAir::Lte {
                idx_len,
                data_len,
                is_less_than_tuple_air,
                is_equal_vec_air,
                ..
            } => {
                let page_main = &builder.partitioned_main()[0].clone();
                let aux_main = &builder.partitioned_main()[1].clone();

                // get the public value x
                let pis = builder.public_values();
                let public_x = pis[..*idx_len].to_vec();

                let local_page = page_main.row_slice(0);
                let local_aux = aux_main.row_slice(0);
                let local_vec = local_page
                    .iter()
                    .chain(local_aux.iter())
                    .cloned()
                    .collect::<Vec<AB::Var>>();
                let local = local_vec.as_slice();

                let local_cols = PageIndexScanInputCols::<AB::Var>::from_slice(
                    local,
                    *idx_len,
                    *data_len,
                    is_less_than_tuple_air.limb_bits().clone(),
                    is_less_than_tuple_air.decomp(),
                    Comp::Lte,
                );

                match local_cols {
                    PageIndexScanInputCols::Lt { .. } => {
                        panic!(
                            "expected PageIndexScanInputCols::Lte, got PageIndexScanInputCols::Lt"
                        );
                    }
                    PageIndexScanInputCols::Lte {
                        is_alloc,
                        idx,
                        x,
                        less_than_x,
                        eq_to_x,
                        satisfies_pred,
                        send_row,
                        is_less_than_tuple_aux,
                        is_equal_vec_aux,
                        ..
                    } => {
                        // here, we are checking if idx <= x
                        let is_less_than_tuple_cols = IsLessThanTupleCols {
                            io: IsLessThanTupleIOCols {
                                x: idx.clone(),
                                y: x.clone(),
                                tuple_less_than: less_than_x,
                            },
                            aux: is_less_than_tuple_aux,
                        };

                        // constrain the indicator that we used to check wheter key < x is correct
                        SubAir::eval(
                            is_less_than_tuple_air,
                            &mut builder.when_transition(),
                            is_less_than_tuple_cols.io,
                            is_less_than_tuple_cols.aux,
                        );

                        // here, we are checking if idx = x
                        let is_equal_vec_cols = IsEqualVecCols {
                            io: IsEqualVecIOCols {
                                x: idx.clone(),
                                y: x.clone(),
                                prod: eq_to_x,
                            },
                            aux: is_equal_vec_aux,
                        };

                        // constrain the indicator that we used to check wheter key = x is correct
                        SubAir::eval(
                            is_equal_vec_air,
                            builder,
                            is_equal_vec_cols.io,
                            is_equal_vec_cols.aux,
                        );

                        // constrain that it satisfies predicate if either less than or equal, and that satisfies is bool
                        builder.assert_eq(less_than_x + eq_to_x, satisfies_pred);
                        builder.assert_bool(satisfies_pred);

                        // constrain that the public value x is the same as the column x
                        for (&local_x, &pub_x) in x.iter().zip(public_x.iter()) {
                            builder.assert_eq(local_x, pub_x);
                        }

                        // constrain that we send the row iff the row is allocated and satisfies the predicate
                        builder.assert_eq(is_alloc * satisfies_pred, send_row);
                        builder.assert_bool(send_row);
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
            }
            PageIndexScanInputAir::Eq {
                idx_len,
                data_len,
                is_equal_vec_air,
                ..
            } => {
                let page_main = &builder.partitioned_main()[0].clone();
                let aux_main = &builder.partitioned_main()[1].clone();

                // get the public value x
                let pis = builder.public_values();
                let public_x = pis[..*idx_len].to_vec();

                let local_page = page_main.row_slice(0);
                let local_aux = aux_main.row_slice(0);
                let local_vec = local_page
                    .iter()
                    .chain(local_aux.iter())
                    .cloned()
                    .collect::<Vec<AB::Var>>();
                let local = local_vec.as_slice();

                let local_cols = PageIndexScanInputCols::<AB::Var>::from_slice(
                    local,
                    *idx_len,
                    *data_len,
                    vec![],
                    0,
                    Comp::Eq,
                );

                match local_cols {
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
                        x,
                        satisfies_pred,
                        send_row,
                        is_equal_vec_aux,
                        ..
                    } => {
                        // here, we are checking if idx = x
                        let is_equal_vec_cols = IsEqualVecCols {
                            io: IsEqualVecIOCols {
                                x: idx,
                                y: x.clone(),
                                prod: satisfies_pred,
                            },
                            aux: is_equal_vec_aux,
                        };

                        // constrain that the public value x is the same as the column x
                        for (&local_x, &pub_x) in x.iter().zip(public_x.iter()) {
                            builder.assert_eq(local_x, pub_x);
                        }

                        // constrain that we send the row iff the row is allocated and satisfies the predicate
                        builder.assert_eq(is_alloc * satisfies_pred, send_row);
                        builder.assert_bool(send_row);

                        // constrain the indicator that we used to check wheter key = x is correct
                        SubAir::eval(
                            is_equal_vec_air,
                            builder,
                            is_equal_vec_cols.io,
                            is_equal_vec_cols.aux,
                        );
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
            }
            PageIndexScanInputAir::Gte {
                idx_len,
                data_len,
                is_less_than_tuple_air,
                is_equal_vec_air,
                ..
            } => {
                let page_main = &builder.partitioned_main()[0].clone();
                let aux_main = &builder.partitioned_main()[1].clone();

                // get the public value x
                let pis = builder.public_values();
                let public_x = pis[..*idx_len].to_vec();

                let local_page = page_main.row_slice(0);
                let local_aux = aux_main.row_slice(0);
                let local_vec = local_page
                    .iter()
                    .chain(local_aux.iter())
                    .cloned()
                    .collect::<Vec<AB::Var>>();
                let local = local_vec.as_slice();

                let local_cols = PageIndexScanInputCols::<AB::Var>::from_slice(
                    local,
                    *idx_len,
                    *data_len,
                    is_less_than_tuple_air.limb_bits().clone(),
                    is_less_than_tuple_air.decomp(),
                    Comp::Gte,
                );

                match local_cols {
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
                        x,
                        greater_than_x,
                        eq_to_x,
                        satisfies_pred,
                        send_row,
                        is_less_than_tuple_aux,
                        is_equal_vec_aux,
                        ..
                    } => {
                        // here, we are checking if idx <= x
                        let is_less_than_tuple_cols = IsLessThanTupleCols {
                            io: IsLessThanTupleIOCols {
                                x: x.clone(),
                                y: idx.clone(),
                                tuple_less_than: greater_than_x,
                            },
                            aux: is_less_than_tuple_aux,
                        };

                        // constrain the indicator that we used to check wheter key < x is correct
                        SubAir::eval(
                            is_less_than_tuple_air,
                            &mut builder.when_transition(),
                            is_less_than_tuple_cols.io,
                            is_less_than_tuple_cols.aux,
                        );

                        // here, we are checking if idx = x
                        let is_equal_vec_cols = IsEqualVecCols {
                            io: IsEqualVecIOCols {
                                x: idx.clone(),
                                y: x.clone(),
                                prod: eq_to_x,
                            },
                            aux: is_equal_vec_aux,
                        };

                        // constrain the indicator that we used to check wheter key = x is correct
                        SubAir::eval(
                            is_equal_vec_air,
                            builder,
                            is_equal_vec_cols.io,
                            is_equal_vec_cols.aux,
                        );

                        // constrain that it satisfies predicate if either less than or equal, and that satisfies is bool
                        builder.assert_eq(greater_than_x + eq_to_x, satisfies_pred);
                        builder.assert_bool(satisfies_pred);

                        // constrain that the public value x is the same as the column x
                        for (&local_x, &pub_x) in x.iter().zip(public_x.iter()) {
                            builder.assert_eq(local_x, pub_x);
                        }

                        // constrain that we send the row iff the row is allocated and satisfies the predicate
                        builder.assert_eq(is_alloc * satisfies_pred, send_row);
                        builder.assert_bool(send_row);
                    }
                    PageIndexScanInputCols::Gt { .. } => {
                        panic!(
                            "expected PageIndexScanInputCols::Gte, got PageIndexScanInputCols::Gt"
                        );
                    }
                }
            }
            PageIndexScanInputAir::Gt {
                idx_len,
                data_len,
                is_less_than_tuple_air,
                ..
            } => {
                let page_main = &builder.partitioned_main()[0].clone();
                let aux_main = &builder.partitioned_main()[1].clone();

                // get the public value x
                let pis = builder.public_values();
                let public_x = pis[..*idx_len].to_vec();

                let local_page = page_main.row_slice(0);
                let local_aux = aux_main.row_slice(0);
                let local_vec = local_page
                    .iter()
                    .chain(local_aux.iter())
                    .cloned()
                    .collect::<Vec<AB::Var>>();
                let local = local_vec.as_slice();

                let local_cols = PageIndexScanInputCols::<AB::Var>::from_slice(
                    local,
                    *idx_len,
                    *data_len,
                    is_less_than_tuple_air.limb_bits().clone(),
                    is_less_than_tuple_air.decomp(),
                    Comp::Gt,
                );

                match local_cols {
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
                        x,
                        satisfies_pred,
                        send_row,
                        is_less_than_tuple_aux,
                        ..
                    } => {
                        // here, we are checking if idx > x
                        let is_less_than_tuple_cols = IsLessThanTupleCols {
                            io: IsLessThanTupleIOCols {
                                x: x.clone(),
                                y: idx,
                                tuple_less_than: satisfies_pred,
                            },
                            aux: is_less_than_tuple_aux,
                        };

                        // constrain that the public value x is the same as the column x
                        for (&local_x, &pub_x) in x.iter().zip(public_x.iter()) {
                            builder.assert_eq(local_x, pub_x);
                        }

                        // constrain that we send the row iff the row is allocated and satisfies the predicate
                        builder.assert_eq(is_alloc * satisfies_pred, send_row);
                        builder.assert_bool(send_row);

                        // constrain the indicator that we used to check wheter key < x is correct
                        SubAir::eval(
                            is_less_than_tuple_air,
                            &mut builder.when_transition(),
                            is_less_than_tuple_cols.io,
                            is_less_than_tuple_cols.aux,
                        );
                    }
                }
            }
        }
    }
}
