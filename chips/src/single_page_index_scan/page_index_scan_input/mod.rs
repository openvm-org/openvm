use derive_more::Display;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{
    is_equal_vec::{columns::IsEqualVecAuxCols, IsEqualVecAir},
    is_less_than_tuple::{columns::IsLessThanTupleAuxCols, IsLessThanTupleAir},
    range_gate::RangeCheckerGateChip,
};

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[derive(Default, Debug, Display, Clone, Deserialize, Serialize, PartialEq)]
pub enum Comp {
    #[default]
    Lt,
    Lte,
    Eq,
    Gte,
    Gt,
}

pub struct StrictCompAir {
    is_less_than_tuple_air: IsLessThanTupleAir,
}

pub struct NonStrictCompAir {
    is_less_than_tuple_air: IsLessThanTupleAir,
    is_equal_vec_air: IsEqualVecAir,
}

pub struct EqCompAir {
    is_equal_vec_air: IsEqualVecAir,
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

    variant_air: PageIndexScanInputAirVariants,
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
}

/// Given a fixed predicate of the form index OP x, where OP is one of {<, <=, =, >=, >}
/// and x is a private input, the PageIndexScanInputChip implements a chip such that the chip:
///
/// 1. Has public value x and OP given by cmp (Lt, Lte, Eq, Gte, or Gt)
/// 2. Sends all rows of the page that match the predicate index OP x where x is the public value
pub struct PageIndexScanInputChip {
    pub air: PageIndexScanInputAir,
    pub range_checker: Arc<RangeCheckerGateChip>,
    pub cmp: Comp,
}

impl PageIndexScanInputChip {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        page_bus_index: usize,
        idx_len: usize,
        data_len: usize,
        idx_limb_bits: usize,
        decomp: usize,
        range_checker: Arc<RangeCheckerGateChip>,
        cmp: Comp,
    ) -> Self {
        let air = PageIndexScanInputAir::new(
            page_bus_index,
            range_checker.bus_index(),
            idx_len,
            data_len,
            idx_limb_bits,
            decomp,
            cmp.clone(),
        );

        Self {
            air,
            range_checker,
            cmp,
        }
    }

    pub fn page_width(&self) -> usize {
        1 + self.air.idx_len + self.air.data_len
    }

    pub fn aux_width(&self) -> usize {
        match &self.air.variant_air {
            PageIndexScanInputAirVariants::Lt(StrictCompAir {
                is_less_than_tuple_air,
                ..
            })
            | PageIndexScanInputAirVariants::Gt(StrictCompAir {
                is_less_than_tuple_air,
                ..
            }) => {
                // x, satisfies_pred, send_row, is_less_than_tuple_aux_cols
                self.air.idx_len
                    + 1
                    + 1
                    + IsLessThanTupleAuxCols::<usize>::get_width(
                        is_less_than_tuple_air.limb_bits(),
                        is_less_than_tuple_air.decomp(),
                        self.air.idx_len,
                    )
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
                self.air.idx_len
                    + 1
                    + 1
                    + 1
                    + 1
                    + IsLessThanTupleAuxCols::<usize>::get_width(
                        is_less_than_tuple_air.limb_bits(),
                        is_less_than_tuple_air.decomp(),
                        self.air.idx_len,
                    )
                    + IsEqualVecAuxCols::<usize>::get_width(self.air.idx_len)
            }
            PageIndexScanInputAirVariants::Eq(EqCompAir { .. }) => {
                // x, satisfies_pred, send_row, is_equal_vec_aux_cols
                self.air.idx_len + 1 + 1 + IsEqualVecAuxCols::<usize>::get_width(self.air.idx_len)
            }
        }
    }

    pub fn air_width(&self) -> usize {
        self.page_width() + self.aux_width()
    }
}
