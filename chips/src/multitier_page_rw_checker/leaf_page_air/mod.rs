use getset::Getters;

use crate::{
    is_less_than_tuple::{columns::IsLessThanTupleAuxCols, IsLessThanTupleAir},
    page_rw_checker::{final_page::IndexedPageWriteAir, initial_page::PageReadAir},
};

use super::page_controller::MyLessThanTupleParams;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[derive(Clone, Debug)]
pub(crate) enum PageRwAir {
    Initial(PageReadAir),
    Final(IndexedPageWriteAir),
}

impl PageRwAir {
    pub fn air_width(&self) -> usize {
        match self {
            PageRwAir::Initial(i) => i.air_width(),
            PageRwAir::Final(fin) => fin.air_width(),
        }
    }
}

#[derive(Clone, Getters)]
pub struct LeafPageAir<const COMMITMENT_LEN: usize> {
    // bus to establish connectivity/internode consistency
    #[getset(get = "pub")]
    path_bus_index: usize,
    // bus to send data to other chips
    #[getset(get = "pub")]
    data_bus_index: usize,

    #[getset(get = "pub")]
    page_chip: PageRwAir,
    // parameter telling if this is a leaf chip on the init side or the final side.
    is_less_than_tuple_air: Option<LeafPageSubAirs>,
    is_less_than_tuple_param: MyLessThanTupleParams,
    is_init: bool,
    idx_len: usize,
    data_len: usize,
    air_id: u32,
}

#[derive(Clone)]
pub struct LeafPageSubAirs {
    pub idx_start: IsLessThanTupleAir,
    pub end_idx: IsLessThanTupleAir,
}

#[allow(clippy::too_many_arguments)]
impl<const COMMITMENT_LEN: usize> LeafPageAir<COMMITMENT_LEN> {
    pub fn new(
        path_bus_index: usize,
        data_bus_index: usize,
        is_less_than_tuple_param: MyLessThanTupleParams,
        lt_bus_index: usize,
        idx_len: usize,
        data_len: usize,
        is_init: bool,
        air_id: u32,
    ) -> Self {
        if is_init {
            Self {
                path_bus_index,
                data_bus_index,
                page_chip: PageRwAir::Initial(PageReadAir::new(data_bus_index, idx_len, data_len)),
                idx_len,
                data_len,
                is_init,
                is_less_than_tuple_air: None,
                is_less_than_tuple_param,
                air_id,
            }
        } else {
            Self {
                path_bus_index,
                data_bus_index,
                page_chip: PageRwAir::Final(IndexedPageWriteAir::new(
                    data_bus_index,
                    lt_bus_index,
                    idx_len,
                    data_len,
                    is_less_than_tuple_param.limb_bits,
                    is_less_than_tuple_param.decomp,
                )),
                idx_len,
                data_len,
                is_init,
                is_less_than_tuple_air: Some(LeafPageSubAirs {
                    idx_start: IsLessThanTupleAir::new(
                        lt_bus_index,
                        vec![is_less_than_tuple_param.limb_bits; idx_len],
                        is_less_than_tuple_param.decomp,
                    ),
                    end_idx: IsLessThanTupleAir::new(
                        lt_bus_index,
                        vec![is_less_than_tuple_param.limb_bits; idx_len],
                        is_less_than_tuple_param.decomp,
                    ),
                }),
                is_less_than_tuple_param,
                air_id,
            }
        }
    }

    // if self.is_final, we need to include range data to establish sortedness
    // in particular, for each idx, prove the idx lies in the start and end.
    // we then need extra columns that contain results of is_less_than comparisons
    // in particular, we need to constrain that is_alloc * ((1 - (idx < start)) * (1 - (end < idx)) - 1) = 0
    pub fn air_width(&self) -> usize {
        1 + self.page_chip().air_width()
            + COMMITMENT_LEN                // own_commitment
            + (1 - self.is_init as usize)
                * (2 * self.idx_len
                    + 2
                    + 2 * IsLessThanTupleAuxCols::<usize>::get_width(
                        vec![self.is_less_than_tuple_param.limb_bits; self.idx_len],
                        self.is_less_than_tuple_param.decomp,
                        self.idx_len,
                    ))
    }
}
