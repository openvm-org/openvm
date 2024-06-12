use afs_stark_backend::air_builders::PartitionedAirBuilder;
use p3_air::{Air, AirBuilderWithPublicValues, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use crate::{
    is_equal_vec::columns::{IsEqualVecAuxCols, IsEqualVecCols, IsEqualVecIOCols},
    is_less_than_tuple::columns::{
        IsLessThanTupleAuxCols, IsLessThanTupleCols, IsLessThanTupleIOCols,
    },
    sub_chip::{AirConfig, SubAir},
};

use super::{
    columns::{
        EqCompAuxCols, NonStrictCompAuxCols, PageIndexScanInputAuxCols, PageIndexScanInputCols,
        StrictCompAuxCols,
    },
    Comp, EqCompAir, NonStrictCompAir, PageIndexScanInputAir, PageIndexScanInputAirVariants,
    StrictCompAir,
};

impl AirConfig for PageIndexScanInputAir {
    type Cols<T> = PageIndexScanInputCols<T>;
}

impl<F: Field> BaseAir<F> for PageIndexScanInputAir {
    fn width(&self) -> usize {
        match &self.subair {
            PageIndexScanInputAirVariants::Lt(StrictCompAir {
                is_less_than_tuple_air,
                ..
            })
            | PageIndexScanInputAirVariants::Gt(StrictCompAir {
                is_less_than_tuple_air,
                ..
            }) => PageIndexScanInputCols::<F>::get_width(
                self.idx_len,
                self.data_len,
                is_less_than_tuple_air.limb_bits().clone(),
                is_less_than_tuple_air.decomp(),
                Comp::Lt,
            ),
            PageIndexScanInputAirVariants::Lte(NonStrictCompAir {
                is_less_than_tuple_air,
                ..
            })
            | PageIndexScanInputAirVariants::Gte(NonStrictCompAir {
                is_less_than_tuple_air,
                ..
            }) => PageIndexScanInputCols::<F>::get_width(
                self.idx_len,
                self.data_len,
                is_less_than_tuple_air.limb_bits().clone(),
                is_less_than_tuple_air.decomp(),
                Comp::Lte,
            ),
            PageIndexScanInputAirVariants::Eq(EqCompAir { .. }) => {
                PageIndexScanInputCols::<F>::get_width(
                    self.idx_len,
                    self.data_len,
                    vec![],
                    0,
                    Comp::Eq,
                )
            }
        }
    }
}

impl<AB: PartitionedAirBuilder + AirBuilderWithPublicValues> Air<AB> for PageIndexScanInputAir
where
    AB::M: Clone,
{
    fn eval(&self, builder: &mut AB) {
        let page_main = &builder.partitioned_main()[0].clone();
        let aux_main = &builder.partitioned_main()[1].clone();

        // get the public value x
        let pis = builder.public_values();
        let public_x = pis[..self.idx_len].to_vec();

        let local_page = page_main.row_slice(0);
        let local_aux = aux_main.row_slice(0);
        let local_vec = local_page
            .iter()
            .chain(local_aux.iter())
            .cloned()
            .collect::<Vec<AB::Var>>();
        let local = local_vec.as_slice();

        let (idx_limb_bits, decomp) = match &self.subair {
            PageIndexScanInputAirVariants::Lt(StrictCompAir {
                is_less_than_tuple_air,
                ..
            })
            | PageIndexScanInputAirVariants::Gt(StrictCompAir {
                is_less_than_tuple_air,
                ..
            })
            | PageIndexScanInputAirVariants::Lte(NonStrictCompAir {
                is_less_than_tuple_air,
                ..
            })
            | PageIndexScanInputAirVariants::Gte(NonStrictCompAir {
                is_less_than_tuple_air,
                ..
            }) => (
                is_less_than_tuple_air.limb_bits(),
                is_less_than_tuple_air.decomp(),
            ),
            PageIndexScanInputAirVariants::Eq(EqCompAir { .. }) => (vec![], 0),
        };

        let cmp = match &self.subair {
            PageIndexScanInputAirVariants::Lt(..) => Comp::Lt,
            PageIndexScanInputAirVariants::Gt(..) => Comp::Gt,
            PageIndexScanInputAirVariants::Lte(..) => Comp::Lte,
            PageIndexScanInputAirVariants::Gte(..) => Comp::Gte,
            PageIndexScanInputAirVariants::Eq(..) => Comp::Eq,
        };

        let local_cols = PageIndexScanInputCols::<AB::Var>::from_slice(
            local,
            self.idx_len,
            self.data_len,
            idx_limb_bits.clone(),
            decomp,
            cmp,
        );

        // constrain that the public value x is the same as the column x
        for (&local_x, &pub_x) in local_cols.x.iter().zip(public_x.iter()) {
            builder.assert_eq(local_x, pub_x);
        }
        // constrain that we send the row iff the row is allocated and satisfies the predicate
        builder.assert_eq(
            local_cols.page_cols.is_alloc * local_cols.satisfies_pred,
            local_cols.send_row,
        );
        // constrain that satisfies_pred and send_row are boolean indicators
        builder.assert_bool(local_cols.satisfies_pred);
        builder.assert_bool(local_cols.send_row);

        let is_less_than_tuple_aux_flattened = match &local_cols.aux_cols {
            PageIndexScanInputAuxCols::Lt(StrictCompAuxCols {
                is_less_than_tuple_aux,
                ..
            })
            | PageIndexScanInputAuxCols::Gt(StrictCompAuxCols {
                is_less_than_tuple_aux,
                ..
            })
            | PageIndexScanInputAuxCols::Lte(NonStrictCompAuxCols {
                is_less_than_tuple_aux,
                ..
            })
            | PageIndexScanInputAuxCols::Gte(NonStrictCompAuxCols {
                is_less_than_tuple_aux,
                ..
            }) => is_less_than_tuple_aux.flatten(),
            PageIndexScanInputAuxCols::Eq(EqCompAuxCols { .. }) => vec![],
        };

        let is_equal_vec_aux_flattened = match &local_cols.aux_cols {
            PageIndexScanInputAuxCols::Eq(EqCompAuxCols {
                is_equal_vec_aux, ..
            })
            | PageIndexScanInputAuxCols::Lte(NonStrictCompAuxCols {
                is_equal_vec_aux, ..
            })
            | PageIndexScanInputAuxCols::Gte(NonStrictCompAuxCols {
                is_equal_vec_aux, ..
            }) => is_equal_vec_aux.flatten(),
            _ => vec![],
        };

        match &self.subair {
            PageIndexScanInputAirVariants::Lt(StrictCompAir {
                is_less_than_tuple_air,
                ..
            }) => {
                // here, we are checking if idx < x
                let is_less_than_tuple_cols = IsLessThanTupleCols {
                    io: IsLessThanTupleIOCols {
                        x: local_cols.page_cols.idx.clone(),
                        y: local_cols.x.clone(),
                        tuple_less_than: local_cols.satisfies_pred,
                    },
                    aux: IsLessThanTupleAuxCols::from_slice(
                        &is_less_than_tuple_aux_flattened,
                        idx_limb_bits.clone(),
                        decomp,
                        self.idx_len,
                    ),
                };

                // constrain the indicator that we used to check whether key < x is correct
                SubAir::eval(
                    is_less_than_tuple_air,
                    &mut builder.when_transition(),
                    is_less_than_tuple_cols.io,
                    is_less_than_tuple_cols.aux,
                );
            }
            PageIndexScanInputAirVariants::Lte(NonStrictCompAir {
                is_less_than_tuple_air,
                is_equal_vec_air,
            }) => {
                match &local_cols.aux_cols {
                    PageIndexScanInputAuxCols::Lte(NonStrictCompAuxCols {
                        satisfies_strict,
                        satisfies_eq,
                        ..
                    }) => {
                        // here, we are checking if idx < x
                        let is_less_than_tuple_cols = IsLessThanTupleCols {
                            io: IsLessThanTupleIOCols {
                                x: local_cols.page_cols.idx.clone(),
                                y: local_cols.x.clone(),
                                tuple_less_than: *satisfies_strict,
                            },
                            aux: IsLessThanTupleAuxCols::from_slice(
                                &is_less_than_tuple_aux_flattened,
                                idx_limb_bits,
                                decomp,
                                self.idx_len,
                            ),
                        };

                        // constrain the indicator that we used to check whether idx < x is correct
                        SubAir::eval(
                            is_less_than_tuple_air,
                            &mut builder.when_transition(),
                            is_less_than_tuple_cols.io,
                            is_less_than_tuple_cols.aux,
                        );

                        // here, we are checking if idx = x
                        let is_equal_vec_cols = IsEqualVecCols {
                            io: IsEqualVecIOCols {
                                x: local_cols.page_cols.idx.clone(),
                                y: local_cols.x.clone(),
                                prod: *satisfies_eq,
                            },
                            aux: IsEqualVecAuxCols::from_slice(
                                &is_equal_vec_aux_flattened,
                                self.idx_len,
                            ),
                        };

                        // constrain the indicator that we used to check whether idx = x is correct
                        SubAir::eval(
                            is_equal_vec_air,
                            builder,
                            is_equal_vec_cols.io,
                            is_equal_vec_cols.aux,
                        );

                        // constrain that satisfies_pred indicates whether idx <= x
                        builder.assert_eq(
                            *satisfies_strict + *satisfies_eq,
                            local_cols.satisfies_pred,
                        );
                    }
                    _ => panic!("Unexpected aux cols"),
                }
            }
            PageIndexScanInputAirVariants::Eq(EqCompAir { is_equal_vec_air }) => {
                // here, we are checking if idx = x
                let is_equal_vec_cols = IsEqualVecCols {
                    io: IsEqualVecIOCols {
                        x: local_cols.page_cols.idx.clone(),
                        y: local_cols.x.clone(),
                        prod: local_cols.satisfies_pred,
                    },
                    aux: IsEqualVecAuxCols::from_slice(&is_equal_vec_aux_flattened, self.idx_len),
                };

                // constrain the indicator that we used to check whether idx = x is correct
                SubAir::eval(
                    is_equal_vec_air,
                    builder,
                    is_equal_vec_cols.io,
                    is_equal_vec_cols.aux,
                );
            }
            PageIndexScanInputAirVariants::Gte(NonStrictCompAir {
                is_less_than_tuple_air,
                is_equal_vec_air,
            }) => {
                match &local_cols.aux_cols {
                    PageIndexScanInputAuxCols::Gte(NonStrictCompAuxCols {
                        satisfies_strict,
                        satisfies_eq,
                        ..
                    }) => {
                        // here, we are checking if idx > x
                        let is_less_than_tuple_cols = IsLessThanTupleCols {
                            io: IsLessThanTupleIOCols {
                                x: local_cols.x.clone(),
                                y: local_cols.page_cols.idx.clone(),
                                tuple_less_than: *satisfies_strict,
                            },
                            aux: IsLessThanTupleAuxCols::from_slice(
                                &is_less_than_tuple_aux_flattened,
                                idx_limb_bits,
                                decomp,
                                self.idx_len,
                            ),
                        };

                        // constrain the indicator that we used to check whether idx > x is correct
                        SubAir::eval(
                            is_less_than_tuple_air,
                            &mut builder.when_transition(),
                            is_less_than_tuple_cols.io,
                            is_less_than_tuple_cols.aux,
                        );

                        // here, we are checking if idx = x
                        let is_equal_vec_cols = IsEqualVecCols {
                            io: IsEqualVecIOCols {
                                x: local_cols.page_cols.idx.clone(),
                                y: local_cols.x.clone(),
                                prod: *satisfies_eq,
                            },
                            aux: IsEqualVecAuxCols::from_slice(
                                &is_equal_vec_aux_flattened,
                                self.idx_len,
                            ),
                        };

                        // constrain the indicator that we used to check whether idx = x is correct
                        SubAir::eval(
                            is_equal_vec_air,
                            builder,
                            is_equal_vec_cols.io,
                            is_equal_vec_cols.aux,
                        );

                        builder.assert_eq(
                            *satisfies_strict + *satisfies_eq,
                            local_cols.satisfies_pred,
                        );
                        builder.assert_bool(local_cols.satisfies_pred);
                    }
                    _ => panic!("Unexpected aux cols"),
                }
            }
            PageIndexScanInputAirVariants::Gt(StrictCompAir {
                is_less_than_tuple_air,
                ..
            }) => {
                // here, we are checking if idx > x
                let is_less_than_tuple_cols = IsLessThanTupleCols {
                    io: IsLessThanTupleIOCols {
                        x: local_cols.x.clone(),
                        y: local_cols.page_cols.idx.clone(),
                        tuple_less_than: local_cols.satisfies_pred,
                    },
                    aux: IsLessThanTupleAuxCols::from_slice(
                        &is_less_than_tuple_aux_flattened,
                        idx_limb_bits,
                        decomp,
                        self.idx_len,
                    ),
                };

                // constrain the indicator that we used to check whether idx > x is correct
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
