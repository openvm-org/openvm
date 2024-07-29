use getset::Getters;

use afs_primitives::{
    is_less_than_tuple::{columns::IsLessThanTupleAuxCols, IsLessThanTupleAir},
    is_zero::IsZeroAir,
};

use super::page_controller::MyLessThanTupleParams;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[derive(Clone)]
pub struct InternalPageSubAirs {
    pub idx1_start: IsLessThanTupleAir,
    pub end_idx2: IsLessThanTupleAir,
    pub idx2_idx1: IsLessThanTupleAir,
    pub idx2_next: IsLessThanTupleAir,
    pub mult_is_1: IsZeroAir,
}

#[derive(Clone, Getters)]
pub struct InternalPageAir<const COMMITMENT_LEN: usize> {
    // bus to establish connectivity/internode consistency
    #[getset(get = "pub")]
    path_bus_index: usize,
    // bus to send data to other chips
    #[getset(get = "pub")]
    data_bus_index: usize,
    // parameter telling if this is a leaf chip on the init side or the final side.
    is_less_than_tuple_air: Option<InternalPageSubAirs>,
    is_less_than_tuple_param: MyLessThanTupleParams,
    is_init: bool,
    idx_len: usize,
    air_id: u32,
}

impl<const COMMITMENT_LEN: usize> InternalPageAir<COMMITMENT_LEN> {
    pub fn new(
        path_bus_index: usize,
        data_bus_index: usize,
        is_less_than_tuple_param: MyLessThanTupleParams,
        lt_bus_index: usize,
        idx_len: usize,
        is_init: bool,
        air_id: u32,
    ) -> Self {
        let subairs = if is_init {
            None
        } else {
            let air = IsLessThanTupleAir::new(
                lt_bus_index,
                vec![is_less_than_tuple_param.limb_bits; idx_len],
                is_less_than_tuple_param.decomp,
            );
            Some(InternalPageSubAirs {
                idx1_start: air.clone(),
                end_idx2: air.clone(),
                idx2_idx1: air.clone(),
                idx2_next: air,
                mult_is_1: IsZeroAir {},
            })
        };
        Self {
            path_bus_index,
            data_bus_index,
            idx_len,
            is_init,
            is_less_than_tuple_param,
            is_less_than_tuple_air: subairs,
            air_id,
        }
    }

    // if self.is_final, we need to include range data to establish sortedness
    // in particular, for each idx, prove the idx lies in the start and end.
    // we then need extra columns that contain results of is_less_than comparisons
    // in particular, we need to constrain that is_alloc * ((1 - (idx < start)) * (1 - (end < idx)) - 1) = 0
    // for both indices
    // we must also assert that the ranges are sorted
    pub fn air_width(&self) -> usize {
        8 + 2 * self.idx_len                    // mult stuff and data
            + COMMITMENT_LEN                // child commitment
            + (1 - self.is_init as usize)
                * (2 * self.idx_len             // prove sort + range inclusion columns
                    + 4
                    + 4 * IsLessThanTupleAuxCols::<usize>::width(                  // aux columns
                        &IsLessThanTupleAir::new(
                            0,
                            vec![self.is_less_than_tuple_param.limb_bits; self.idx_len],
                            self.is_less_than_tuple_param.decomp,
                        ),
                    )
                    + 1) // is_zero
    }

    pub fn main_width(&self) -> usize {
        6                 // mult stuff
            + (1 - self.is_init as usize)
                * (2 * self.idx_len             // prove sort + range inclusion columns
                    + 4
                    + 4 * IsLessThanTupleAuxCols::<usize>::width(                  // aux columns
                        &IsLessThanTupleAir::new(
                            0,
                            vec![self.is_less_than_tuple_param.limb_bits; self.idx_len],
                            self.is_less_than_tuple_param.decomp,
                        ),
                    )
                    + 1) // is_zero
    }

    pub fn cached_width(&self) -> usize {
        2 + 2 * self.idx_len + COMMITMENT_LEN
    }
}
