use getset::Getters;

use crate::{
    is_equal::IsEqualChip,
    is_less_than_tuple::{
        columns::{IsLessThanTupleAuxCols, IsLessThanTupleCols},
        IsLessThanTupleAir,
    },
    is_zero::IsZeroChip,
};

use super::page_controller::LessThanTupleParams;

pub mod air;
pub mod chip;
pub mod columns;
pub mod trace;

#[derive(Clone)]
pub struct InternalPageSubAirs {
    pub key1_start: IsLessThanTupleAir,
    pub end_key1: IsLessThanTupleAir,
    pub key2_start: IsLessThanTupleAir,
    pub end_key2: IsLessThanTupleAir,
    pub end_start: IsLessThanTupleAir,
    pub end_next: IsLessThanTupleAir,
    pub mult_is_1: IsZeroChip,
}

#[derive(Clone)]
pub struct InternalPageSubAirBuses {
    pub key1_start: usize,
    pub end_key1: usize,
    pub key2_start: usize,
    pub end_key2: usize,
    pub end_start: usize,
    pub end_next: usize,
}

#[derive(Clone, Getters)]
pub struct InternalPageChip<const COMMITMENT_LEN: usize> {
    // bus to establish connectivity/internode consistency
    #[getset(get = "pub")]
    path_bus_index: usize,
    // bus to send data to other chips
    #[getset(get = "pub")]
    data_bus_index: usize,
    // parameter telling if this is a leaf chip on the init side or the final side.
    is_less_than_tuple_air: Option<InternalPageSubAirs>,
    is_less_than_tuple_param: LessThanTupleParams,
    is_init: bool,
    idx_len: usize,
}

impl<const COMMITMENT_LEN: usize> InternalPageChip<COMMITMENT_LEN> {
    pub fn new(
        path_bus_index: usize,
        data_bus_index: usize,
        is_less_than_tuple_param: LessThanTupleParams,
        lt_bus_index: usize,
        idx_len: usize,
        is_init: bool,
    ) -> Self {
        let subairs = if is_init {
            None
        } else {
            let air = IsLessThanTupleAir::new(
                lt_bus_index,
                is_less_than_tuple_param.range_max, // unsure about this
                is_less_than_tuple_param.limb_bits.clone(),
                is_less_than_tuple_param.decomp,
            );
            Some(InternalPageSubAirs {
                key1_start: air.clone(),
                end_key1: air.clone(),
                key2_start: air.clone(),
                end_key2: air.clone(),
                end_start: air.clone(),
                end_next: air,
                mult_is_1: IsZeroChip {},
            })
        };
        Self {
            path_bus_index,
            data_bus_index,
            idx_len,
            is_init,
            is_less_than_tuple_param,
            is_less_than_tuple_air: subairs,
        }
    }

    // if self.is_final, we need to include range data to establish sortedness
    // in particular, for each idx, prove the idx lies in the start and end.
    // we then need extra columns that contain results of is_less_than comparisons
    // in particular, we need to constrain that is_alloc * ((1 - (idx < start)) * (1 - (end < idx)) - 1) = 0
    pub fn air_width(&self) -> usize {
        4 + 2 * self.idx_len
            + 2 * COMMITMENT_LEN
            + (1 - self.is_init as usize)
                * (3 * self.idx_len
                    + 6
                    + 6 * IsLessThanTupleCols::<usize>::get_width(
                        self.is_less_than_tuple_param.limb_bits.clone(),
                        self.is_less_than_tuple_param.decomp,
                        self.idx_len,
                    )
                    - 12 * self.idx_len
                    - 6
                    + 1)
    }
}
