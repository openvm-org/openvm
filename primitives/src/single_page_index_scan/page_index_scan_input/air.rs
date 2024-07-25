use afs_stark_backend::{air_builders::PartitionedAirBuilder, interaction::InteractionBuilder};
use p3_air::{Air, AirBuilderWithPublicValues, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use crate::{
    is_equal_vec::{
        columns::{IsEqualVecAuxCols, IsEqualVecCols, IsEqualVecIoCols},
        IsEqualVecAir,
    },
    is_less_than_tuple::{
        columns::{IsLessThanTupleAuxCols, IsLessThanTupleCols, IsLessThanTupleIoCols},
        IsLessThanTupleAir,
    },
    sub_chip::{AirConfig, SubAir},
};

use super::{
    columns::{
        EqCompAuxCols, NonStrictCompAuxCols, PageIndexScanInputAuxCols, PageIndexScanInputCols,
        StrictCompAuxCols,
    },
    Comp,
};

pub struct StrictCompAir {
    pub is_less_than_tuple_air: IsLessThanTupleAir,
}

// TODO[optimization]: <= is same as not >
pub struct NonStrictCompAir {
    pub is_less_than_tuple_air: IsLessThanTupleAir,
    pub is_equal_vec_air: IsEqualVecAir,
}

pub struct EqCompAir {
    pub is_equal_vec_air: IsEqualVecAir,
}

pub enum PageIndexScanInputAirVariants {
    Lt(StrictCompAir),
    Lte(NonStrictCompAir),
    Eq(EqCompAir),
    Gte(NonStrictCompAir),
    Gt(StrictCompAir),
}

pub struct PageIndexScanInputAir {
    pub page_bus_index: usize,
    pub idx_len: usize,
    pub data_len: usize,

    pub(super) variant_air: PageIndexScanInputAirVariants,
}

impl PageIndexScanInputAir {
    pub fn new(
        page_bus_index: usize,
        range_bus_index: usize,
        idx_len: usize,
        data_len: usize,
        idx_limb_bits: usize,
        decomp: usize,
        cmp: Comp,
    ) -> Self {
        let is_less_than_tuple_air =
            IsLessThanTupleAir::new(range_bus_index, vec![idx_limb_bits; idx_len], decomp);
        let is_equal_vec_air = IsEqualVecAir::new(idx_len);

        let variant_air = match cmp {
            Comp::Lt => PageIndexScanInputAirVariants::Lt(StrictCompAir {
                is_less_than_tuple_air,
            }),
            Comp::Lte => PageIndexScanInputAirVariants::Lte(NonStrictCompAir {
                is_less_than_tuple_air,
                is_equal_vec_air,
            }),
            Comp::Eq => PageIndexScanInputAirVariants::Eq(EqCompAir { is_equal_vec_air }),
            Comp::Gte => PageIndexScanInputAirVariants::Gte(NonStrictCompAir {
                is_less_than_tuple_air,
                is_equal_vec_air,
            }),
            Comp::Gt => PageIndexScanInputAirVariants::Gt(StrictCompAir {
                is_less_than_tuple_air,
            }),
        };

        Self {
            page_bus_index,
            idx_len,
            data_len,
            variant_air,
        }
    }

    pub fn page_width(&self) -> usize {
        1 + self.idx_len + self.data_len
    }

    pub fn aux_width(&self) -> usize {
        match &self.variant_air {
            PageIndexScanInputAirVariants::Lt(StrictCompAir {
                is_less_than_tuple_air,
                ..
            })
            | PageIndexScanInputAirVariants::Gt(StrictCompAir {
                is_less_than_tuple_air,
                ..
            }) => {
                // x, satisfies_pred, send_row, is_less_than_tuple_aux_cols
                self.idx_len
                    + 1
                    + 1
                    + IsLessThanTupleAuxCols::<usize>::width(is_less_than_tuple_air)
            }
            PageIndexScanInputAirVariants::Lte(NonStrictCompAir {
                is_less_than_tuple_air,
                ..
            })
            | PageIndexScanInputAirVariants::Gte(NonStrictCompAir {
                is_less_than_tuple_air,
                ..
            }) => {
                // x, satisfies_pred, send_row, satisfies_strict_comp, satisfies_eq_comp,
                // is_less_than_tuple_aux_cols, is_equal_vec_aux_cols
                self.idx_len
                    + 1
                    + 1
                    + 1
                    + 1
                    + IsLessThanTupleAuxCols::<usize>::width(is_less_than_tuple_air)
                    + IsEqualVecAuxCols::<usize>::width(self.idx_len)
            }
            PageIndexScanInputAirVariants::Eq(EqCompAir { .. }) => {
                // x, satisfies_pred, send_row, is_equal_vec_aux_cols
                self.idx_len + 1 + 1 + IsEqualVecAuxCols::<usize>::width(self.idx_len)
            }
        }
    }

    pub fn air_width(&self) -> usize {
        self.page_width() + self.aux_width()
    }
}

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
                &is_less_than_tuple_air.limb_bits,
                is_less_than_tuple_air.decomp,
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
                &is_less_than_tuple_air.limb_bits,
                is_less_than_tuple_air.decomp,
                Comp::Lte,
            ),
            PageIndexScanInputAirVariants::Eq(EqCompAir { .. }) => {
                // since get_width doesn't use idx_limb_bits and decomp for when comparator is =, we can pass in dummy values
                PageIndexScanInputCols::<F>::get_width(
                    self.idx_len,
                    self.data_len,
                    &[],
                    0,
                    Comp::Eq,
                )
            }
        }
    }
}

impl<AB> Air<AB> for PageIndexScanInputAir
where
    AB: PartitionedAirBuilder + AirBuilderWithPublicValues + InteractionBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let page_main = &builder.partitioned_main()[0];
        let aux_main = &builder.partitioned_main()[1];

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
                &is_less_than_tuple_air.limb_bits,
                is_less_than_tuple_air.decomp,
            ),
            PageIndexScanInputAirVariants::Eq(EqCompAir { .. }) => (&vec![], 0),
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
            idx_limb_bits,
            decomp,
            cmp,
        );
        drop(local_page);
        drop(local_aux);

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
        self.eval_interactions(builder, page_cols.idx, page_cols.data, local_cols.send_row);
    }
}
