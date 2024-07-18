use afs_stark_backend::air_builders::PartitionedAirBuilder;
use p3_air::{Air, AirBuilderWithPublicValues, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use crate::{
    is_equal_vec::columns::{IsEqualVecCols, IsEqualVecIoCols},
    is_less_than_tuple::columns::{IsLessThanTupleCols, IsLessThanTupleIoCols},
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
        match &self.variant_air {
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
                // since get_width doesn't use idx_limb_bits and decomp for when comparator is =, we can pass in dummy values
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

        // get the idx_limb_bits and decomp, which will be used to generate local_cols
        let (idx_limb_bits, decomp) = match &self.variant_air {
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

        // get the comparator
        let cmp = match &self.variant_air {
            PageIndexScanInputAirVariants::Lt(..) => Comp::Lt,
            PageIndexScanInputAirVariants::Gt(..) => Comp::Gt,
            PageIndexScanInputAirVariants::Lte(..) => Comp::Lte,
            PageIndexScanInputAirVariants::Gte(..) => Comp::Gte,
            PageIndexScanInputAirVariants::Eq(..) => Comp::Eq,
        };

        let PageIndexScanInputCols {
            page_cols,
            local_cols,
        } = PageIndexScanInputCols::<AB::Var>::from_partitioned_slice(
            &local_page,
            &local_aux,
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
            page_cols.is_alloc * local_cols.satisfies_pred,
            local_cols.send_row,
        );
        // constrain that satisfies_pred and send_row are boolean indicators
        builder.assert_bool(local_cols.satisfies_pred);
        builder.assert_bool(local_cols.send_row);

        // get the indicators for strict and equal comparisons
        let (strict_comp_ind, equal_comp_ind): (Option<AB::Var>, Option<AB::Var>) =
            match &local_cols.aux_cols {
                PageIndexScanInputAuxCols::Lt(..) | PageIndexScanInputAuxCols::Gt(..) => {
                    (Some(local_cols.satisfies_pred), None)
                }
                PageIndexScanInputAuxCols::Lte(NonStrictCompAuxCols {
                    satisfies_strict_comp,
                    satisfies_eq_comp,
                    ..
                })
                | PageIndexScanInputAuxCols::Gte(NonStrictCompAuxCols {
                    satisfies_strict_comp,
                    satisfies_eq_comp,
                    ..
                }) => (Some(*satisfies_strict_comp), Some(*satisfies_eq_comp)),
                PageIndexScanInputAuxCols::Eq(..) => (None, Some(local_cols.satisfies_pred)),
            };

        // generate aux columns for IsLessThanTuple
        let is_less_than_tuple_cols: Option<IsLessThanTupleCols<AB::Var>> =
            match &local_cols.aux_cols {
                PageIndexScanInputAuxCols::Lt(StrictCompAuxCols {
                    is_less_than_tuple_aux,
                    ..
                })
                | PageIndexScanInputAuxCols::Lte(NonStrictCompAuxCols {
                    is_less_than_tuple_aux,
                    ..
                }) => Some(IsLessThanTupleCols {
                    io: IsLessThanTupleIoCols {
                        // idx < x
                        x: page_cols.idx.clone(),
                        y: local_cols.x.clone(),
                        // use the strict_comp_ind
                        tuple_less_than: strict_comp_ind.unwrap(),
                    },
                    aux: is_less_than_tuple_aux.clone(),
                }),
                PageIndexScanInputAuxCols::Gt(StrictCompAuxCols {
                    is_less_than_tuple_aux,
                    ..
                })
                | PageIndexScanInputAuxCols::Gte(NonStrictCompAuxCols {
                    is_less_than_tuple_aux,
                    ..
                }) => Some(IsLessThanTupleCols {
                    io: IsLessThanTupleIoCols {
                        // idx > x
                        x: local_cols.x.clone(),
                        y: page_cols.idx.clone(),
                        // use the strict_comp_ind
                        tuple_less_than: strict_comp_ind.unwrap(),
                    },
                    aux: is_less_than_tuple_aux.clone(),
                }),
                PageIndexScanInputAuxCols::Eq(EqCompAuxCols { .. }) => None,
            };

        // generate aux columns for IsEqualVec
        let is_equal_vec_cols: Option<IsEqualVecCols<AB::Var>> = match &local_cols.aux_cols {
            PageIndexScanInputAuxCols::Eq(EqCompAuxCols {
                is_equal_vec_aux, ..
            })
            | PageIndexScanInputAuxCols::Lte(NonStrictCompAuxCols {
                is_equal_vec_aux, ..
            })
            | PageIndexScanInputAuxCols::Gte(NonStrictCompAuxCols {
                is_equal_vec_aux, ..
            }) => {
                let is_equal_vec_cols = IsEqualVecCols {
                    io: IsEqualVecIoCols {
                        x: page_cols.idx.clone(),
                        y: local_cols.x.clone(),
                        // use the equal_comp_ind
                        is_equal: equal_comp_ind.unwrap(),
                    },
                    aux: is_equal_vec_aux.clone(),
                };
                Some(is_equal_vec_cols)
            }
            _ => None,
        };

        // constrain that satisfies pred is correct
        match &self.variant_air {
            PageIndexScanInputAirVariants::Lt(StrictCompAir {
                is_less_than_tuple_air,
                ..
            })
            | PageIndexScanInputAirVariants::Gt(StrictCompAir {
                is_less_than_tuple_air,
                ..
            }) => {
                let is_less_than_tuple_cols = is_less_than_tuple_cols.unwrap();

                // constrain the indicator that we used to check the strict comp is correct
                SubAir::eval(
                    is_less_than_tuple_air,
                    builder,
                    is_less_than_tuple_cols.io,
                    is_less_than_tuple_cols.aux,
                );
            }
            PageIndexScanInputAirVariants::Lte(NonStrictCompAir {
                is_less_than_tuple_air,
                is_equal_vec_air,
            })
            | PageIndexScanInputAirVariants::Gte(NonStrictCompAir {
                is_less_than_tuple_air,
                is_equal_vec_air,
            }) => {
                let is_less_than_tuple_cols = is_less_than_tuple_cols.unwrap();
                let is_equal_vec_cols = is_equal_vec_cols.unwrap();

                // constrain the indicator that we used to check the strict comp is correct
                SubAir::eval(
                    is_less_than_tuple_air,
                    builder,
                    is_less_than_tuple_cols.io,
                    is_less_than_tuple_cols.aux,
                );

                // constrain the indicator that we used to check for equality is correct
                SubAir::eval(
                    is_equal_vec_air,
                    builder,
                    is_equal_vec_cols.io,
                    is_equal_vec_cols.aux,
                );

                // constrain that satisfies_pred indicates the nonstrict comparison
                builder.assert_eq(
                    strict_comp_ind.unwrap() + equal_comp_ind.unwrap(),
                    local_cols.satisfies_pred,
                );
            }
            PageIndexScanInputAirVariants::Eq(EqCompAir { is_equal_vec_air }) => {
                let is_equal_vec_cols = is_equal_vec_cols.unwrap();

                // constrain the indicator that we used to check whether idx = x is correct
                SubAir::eval(
                    is_equal_vec_air,
                    builder,
                    is_equal_vec_cols.io,
                    is_equal_vec_cols.aux,
                );
            }
        }
    }
}
