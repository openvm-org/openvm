use afs_primitives::{
    is_equal_vec::{columns::IsEqualVecAuxCols, IsEqualVecAir},
    is_less_than_tuple::{columns::IsLessThanTupleAuxCols, IsLessThanTupleAir},
};
use p3_air::BaseAir;
use p3_field::Field;

use crate::common::comp::{
    air::{EqCompAir, StrictCompAir},
    Comp,
};

impl<F: Field> BaseAir<F> for FilterInputTableAir {
    fn width(&self) -> usize {
        self.air_width()
    }
}

#[derive(derive_new::new)]
pub enum FilterAirVariants {
    Lt(StrictCompAir),
    Lte(StrictCompAir),
    Eq(EqCompAir),
    Gte(StrictCompAir),
    Gt(StrictCompAir),
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
        let is_less_than_tuple_air =
            IsLessThanTupleAir::new(range_bus_index, vec![idx_limb_bits; idx_len], decomp);
        let is_equal_vec_air = IsEqualVecAir::new(idx_len);
        let variant_air = match cmp {
            Comp::Lt => FilterAirVariants::Lt(StrictCompAir {
                is_less_than_tuple_air,
            }),
            Comp::Lte => FilterAirVariants::Lte(StrictCompAir {
                is_less_than_tuple_air,
            }),
            Comp::Eq => FilterAirVariants::Eq(EqCompAir { is_equal_vec_air }),
            Comp::Gte => FilterAirVariants::Gte(StrictCompAir {
                is_less_than_tuple_air,
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
            FilterAirVariants::Lt(StrictCompAir {
                is_less_than_tuple_air,
                ..
            })
            | FilterAirVariants::Lte(StrictCompAir {
                is_less_than_tuple_air,
                ..
            })
            | FilterAirVariants::Gt(StrictCompAir {
                is_less_than_tuple_air,
                ..
            })
            | FilterAirVariants::Gte(StrictCompAir {
                is_less_than_tuple_air,
                ..
            }) => {
                // x, satisfies_pred, send_row, is_less_than_tuple_aux_cols
                self.idx_len
                    + 1
                    + 1
                    + IsLessThanTupleAuxCols::<usize>::width(is_less_than_tuple_air)
            }
            FilterAirVariants::Eq(EqCompAir { .. }) => {
                // x, satisfies_pred, send_row, is_equal_vec_aux_cols
                self.idx_len + 1 + 1 + IsEqualVecAuxCols::<usize>::width(self.idx_len)
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
