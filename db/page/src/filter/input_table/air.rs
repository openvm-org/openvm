use afs_primitives::{
    is_equal_vec::{
        columns::{IsEqualVecAuxCols, IsEqualVecCols, IsEqualVecIoCols},
        IsEqualVecAir,
    },
    is_less_than_tuple::{
        columns::{IsLessThanTupleAuxCols, IsLessThanTupleCols, IsLessThanTupleIoCols},
        IsLessThanTupleAir,
    },
    sub_chip::SubAir,
};
use afs_stark_backend::{air_builders::PartitionedAirBuilder, interaction::InteractionBuilder};
use p3_air::{Air, AirBuilderWithPublicValues, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::columns::{FilterInputCols, FilterInputTableAuxCols};
use crate::common::comp::{
    air::{EqCompAir, StrictCompAir, StrictInvCompAir},
    Comp,
};

#[derive(derive_new::new)]
pub enum FilterAirVariants {
    Lt(StrictCompAir),
    Lte(StrictInvCompAir),
    Gt(StrictCompAir),
    Gte(StrictInvCompAir),
    Eq(EqCompAir),
}

pub struct FilterInputTableAir {
    pub page_bus_index: usize,
    pub idx_len: usize,
    pub data_len: usize,
    pub start_col: usize,
    pub end_col: usize,
    pub(super) variant_air: FilterAirVariants,
}

impl FilterInputTableAir {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        page_bus_index: usize,
        range_bus_index: usize,
        idx_len: usize,
        data_len: usize,
        start_col: usize,
        end_col: usize,
        idx_limb_bits: usize,
        decomp: usize,
        cmp: Comp,
    ) -> Self {
        let select_len = end_col - start_col;
        let is_less_than_tuple_air =
            IsLessThanTupleAir::new(range_bus_index, vec![idx_limb_bits; select_len], decomp);
        let is_equal_vec_air = IsEqualVecAir::new(select_len);
        let variant_air = match cmp {
            Comp::Lt => FilterAirVariants::Lt(StrictCompAir {
                is_less_than_tuple_air,
            }),
            Comp::Lte => FilterAirVariants::Lte(StrictInvCompAir {
                is_less_than_tuple_air,
                inv: 1,
            }),
            Comp::Eq => FilterAirVariants::Eq(EqCompAir { is_equal_vec_air }),
            Comp::Gte => FilterAirVariants::Gte(StrictInvCompAir {
                is_less_than_tuple_air,
                inv: 1,
            }),
            Comp::Gt => FilterAirVariants::Gt(StrictCompAir {
                is_less_than_tuple_air,
            }),
        };

        Self {
            page_bus_index,
            idx_len,
            data_len,
            start_col,
            end_col,
            variant_air,
        }
    }

    pub fn table_width(&self) -> usize {
        1 + self.idx_len + self.data_len
    }

    pub fn aux_width(&self) -> usize {
        match &self.variant_air {
            FilterAirVariants::Lt(strict_comp_air) | FilterAirVariants::Gt(strict_comp_air) => {
                // x, satisfies_pred, send_row, is_less_than_tuple_aux_cols
                self.num_filter_cols()
                    + 1
                    + 1
                    + IsLessThanTupleAuxCols::<usize>::width(
                        &strict_comp_air.is_less_than_tuple_air,
                    )
            }
            FilterAirVariants::Lte(strict_inv_comp_air)
            | FilterAirVariants::Gte(strict_inv_comp_air) => {
                // x, satisfies_pred, send_row, inv, is_less_than_tuple_aux_cols
                self.num_filter_cols()
                    + 1
                    + 1
                    + 1
                    + IsLessThanTupleAuxCols::<usize>::width(
                        &strict_inv_comp_air.is_less_than_tuple_air,
                    )
            }
            FilterAirVariants::Eq(_) => {
                // x, satisfies_pred, send_row, is_equal_vec_aux_cols
                self.num_filter_cols()
                    + 1
                    + 1
                    + IsEqualVecAuxCols::<usize>::width(self.num_filter_cols())
            }
        }
    }

    pub fn start_filter_col(&self) -> usize {
        self.start_col
    }

    pub fn num_filter_cols(&self) -> usize {
        self.end_col - self.start_col
    }

    pub fn air_width(&self) -> usize {
        self.table_width() + self.aux_width()
    }
}

impl<F: Field> BaseAir<F> for FilterInputTableAir {
    fn width(&self) -> usize {
        match &self.variant_air {
            FilterAirVariants::Lt(strict_comp_air) | FilterAirVariants::Gt(strict_comp_air) => {
                FilterInputCols::<F>::get_width(
                    self.idx_len,
                    self.data_len,
                    self.start_col,
                    self.end_col,
                    &strict_comp_air.is_less_than_tuple_air.limb_bits,
                    strict_comp_air.is_less_than_tuple_air.decomp,
                    Comp::Lt,
                )
            }
            FilterAirVariants::Lte(strict_inv_comp_air)
            | FilterAirVariants::Gte(strict_inv_comp_air) => FilterInputCols::<F>::get_width(
                self.idx_len,
                self.data_len,
                self.start_col,
                self.end_col,
                &strict_inv_comp_air.is_less_than_tuple_air.limb_bits,
                strict_inv_comp_air.is_less_than_tuple_air.decomp,
                Comp::Lt,
            ),
            FilterAirVariants::Eq(_) => FilterInputCols::<F>::get_width(
                self.idx_len,
                self.data_len,
                self.start_col,
                self.end_col,
                &[],
                0,
                Comp::Eq,
            ),
        }
    }
}

impl<AB> Air<AB> for FilterInputTableAir
where
    AB: PartitionedAirBuilder + AirBuilderWithPublicValues + InteractionBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let page_main = &builder.partitioned_main()[0];
        let aux_main = &builder.partitioned_main()[1];

        // get the public value x
        let pis = builder.public_values();
        let select_len = self.end_col - self.start_col;
        let public_x = pis[..select_len].to_vec();

        let local_page = page_main.row_slice(0);
        let local_aux = aux_main.row_slice(0);

        // Get limb bits and decomp to generate local cols later
        let (limb_bits, decomp) = match &self.variant_air {
            FilterAirVariants::Lt(strict_comp_air) | FilterAirVariants::Gt(strict_comp_air) => (
                &strict_comp_air.is_less_than_tuple_air.limb_bits,
                strict_comp_air.is_less_than_tuple_air.decomp,
            ),
            FilterAirVariants::Lte(strict_inv_comp_air)
            | FilterAirVariants::Gte(strict_inv_comp_air) => (
                &strict_inv_comp_air.is_less_than_tuple_air.limb_bits,
                strict_inv_comp_air.is_less_than_tuple_air.decomp,
            ),
            FilterAirVariants::Eq(_) => (&vec![], 0),
        };

        // Get the comparator
        let cmp = match &self.variant_air {
            FilterAirVariants::Lt(..) => Comp::Lt,
            FilterAirVariants::Lte(..) => Comp::Lte,
            FilterAirVariants::Eq(..) => Comp::Eq,
            FilterAirVariants::Gte(..) => Comp::Gte,
            FilterAirVariants::Gt(..) => Comp::Gt,
        };

        let FilterInputCols {
            page_cols,
            local_cols,
            start_col,
            end_col,
        } = FilterInputCols::from_partitioned_slice(
            &local_page,
            &local_aux,
            self.idx_len,
            self.data_len,
            self.start_col,
            self.end_col,
            limb_bits,
            decomp,
            cmp,
        );
        drop(local_page);
        drop(local_aux);

        // Get the selected columns
        let select_cols = page_cols.to_vec();
        let select_cols = select_cols[start_col + 1..end_col + 1].to_vec();

        // Constrain that the public value x is the same as the column x
        for (&local_x, &pub_x) in local_cols.x.iter().zip(public_x.iter()) {
            builder.assert_eq(local_x, pub_x);
        }

        // Constrain that we send the row iff the row is allocated and satisfies the predicate
        builder.assert_eq(
            page_cols.is_alloc * local_cols.satisfies_pred,
            local_cols.send_row,
        );

        // Constrain that satisfies_pred and send_row are boolean indicators
        builder.assert_bool(local_cols.satisfies_pred);
        builder.assert_bool(local_cols.send_row);

        // Get indicators for strict & equal comparisons
        let (strict_comp_ind, strict_comp_ind_inv, equal_comp_ind): (
            Option<AB::Var>,
            Option<AB::Var>,
            Option<AB::Var>,
        ) = match &local_cols.aux_cols {
            FilterInputTableAuxCols::Lt(_) | FilterInputTableAuxCols::Gt(_) => {
                (Some(local_cols.satisfies_pred), None, None)
            }
            FilterInputTableAuxCols::Lte(strict_inv_comp_aux)
            | FilterInputTableAuxCols::Gte(strict_inv_comp_aux) => {
                let inv = strict_inv_comp_aux.inv;
                (None, Some(inv), None)
            }
            FilterInputTableAuxCols::Eq(_) => (None, None, Some(local_cols.satisfies_pred)),
        };

        if let Some(inv) = strict_comp_ind_inv {
            builder.assert_bool(inv);
            // TODO: why does satisfies_pred == 1 for LTE/GTE?
            builder.assert_eq(AB::Expr::one() - local_cols.satisfies_pred, inv);
            // builder.assert_one(inv);
            // builder.assert_zero(local_cols.satisfies_pred);
        }

        // Generate aux columns for IsLessThanTuple
        let is_less_than_tuple_cols: Option<IsLessThanTupleCols<AB::Var>> =
            match &local_cols.aux_cols {
                FilterInputTableAuxCols::Lt(strict_aux_cols) => Some(IsLessThanTupleCols {
                    io: IsLessThanTupleIoCols {
                        x: select_cols.clone(),
                        y: local_cols.x.clone(),
                        tuple_less_than: strict_comp_ind.unwrap(),
                    },
                    aux: strict_aux_cols.is_less_than_tuple_aux.clone(),
                }),
                FilterInputTableAuxCols::Gte(strict_inv_aux_cols) => Some(IsLessThanTupleCols {
                    io: IsLessThanTupleIoCols {
                        x: select_cols.clone(),
                        y: local_cols.x.clone(),
                        tuple_less_than: strict_comp_ind_inv.unwrap(),
                    },
                    aux: strict_inv_aux_cols.is_less_than_tuple_aux.clone(),
                }),
                FilterInputTableAuxCols::Gt(strict_aux_cols) => Some(IsLessThanTupleCols {
                    io: IsLessThanTupleIoCols {
                        x: local_cols.x.clone(),
                        y: select_cols.clone(),
                        tuple_less_than: strict_comp_ind.unwrap(),
                    },
                    aux: strict_aux_cols.is_less_than_tuple_aux.clone(),
                }),
                FilterInputTableAuxCols::Lte(strict_inv_aux_cols) => Some(IsLessThanTupleCols {
                    io: IsLessThanTupleIoCols {
                        x: local_cols.x.clone(),
                        y: select_cols.clone(),
                        tuple_less_than: strict_comp_ind_inv.unwrap(),
                    },
                    aux: strict_inv_aux_cols.is_less_than_tuple_aux.clone(),
                }),
                FilterInputTableAuxCols::Eq(_) => None,
            };

        // Generate aux columns for IsEqualVec
        let is_equal_vec_cols: Option<IsEqualVecCols<AB::Var>> = match &local_cols.aux_cols {
            FilterInputTableAuxCols::Eq(eq_aux_cols) => Some(IsEqualVecCols {
                io: IsEqualVecIoCols {
                    x: select_cols.clone(),
                    y: local_cols.x.clone(),
                    is_equal: equal_comp_ind.unwrap(),
                },
                aux: eq_aux_cols.is_equal_vec_aux.clone(),
            }),
            _ => None,
        };

        // Constrain that satisfies pred is correct
        match &self.variant_air {
            FilterAirVariants::Lt(strict_comp_air) | FilterAirVariants::Gt(strict_comp_air) => {
                let is_less_than_tuple_cols = is_less_than_tuple_cols.unwrap();
                SubAir::eval(
                    &strict_comp_air.is_less_than_tuple_air,
                    builder,
                    is_less_than_tuple_cols.io,
                    is_less_than_tuple_cols.aux,
                );
            }
            FilterAirVariants::Lte(strict_inv_comp_air)
            | FilterAirVariants::Gte(strict_inv_comp_air) => {
                let is_less_than_tuple_cols = is_less_than_tuple_cols.unwrap();

                // SubAir::eval(
                //     &strict_inv_comp_air.is_less_than_tuple_air,
                //     builder,
                //     is_less_than_tuple_cols.io,
                //     is_less_than_tuple_cols.aux,
                // );
            }
            FilterAirVariants::Eq(eq_comp_air) => {
                let is_equal_vec_cols = is_equal_vec_cols.unwrap();
                SubAir::eval(
                    &eq_comp_air.is_equal_vec_air,
                    builder,
                    is_equal_vec_cols.io,
                    is_equal_vec_cols.aux,
                );
            }
        }
        self.eval_interactions(builder, page_cols.idx, page_cols.data, local_cols.send_row);
    }
}
