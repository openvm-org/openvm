use crate::is_less_than_tuple::{columns::IsLessThanTupleAuxCols, IsLessThanTupleAir};

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

pub struct IntersectorAir {
    pub t1_intersector_bus_index: usize,
    pub t2_intersector_bus_index: usize,
    pub intersector_t2_bus_index: usize,

    pub idx_len: usize,

    pub lt_chip: IsLessThanTupleAir,
}

impl IntersectorAir {
    pub fn new(
        range_bus_index: usize,
        t1_intersector_bus_index: usize,
        t2_intersector_bus_index: usize,
        intersector_t2_bus_index: usize,
        idx_len: usize,
        idx_limb_bits: usize,
        decomp: usize,
    ) -> Self {
        Self {
            t1_intersector_bus_index,
            t2_intersector_bus_index,
            intersector_t2_bus_index,
            idx_len,
            lt_chip: IsLessThanTupleAir::new(range_bus_index, vec![idx_limb_bits; idx_len], decomp),
        }
    }

    pub fn io_width(&self) -> usize {
        4 + self.idx_len
    }

    pub fn aux_width(&self) -> usize {
        IsLessThanTupleAuxCols::<usize>::get_width(
            self.lt_chip.limb_bits(),
            self.lt_chip.decomp(),
            self.idx_len,
        ) + 1
    }

    pub fn air_width(&self) -> usize {
        self.io_width() + self.aux_width()
    }
}
