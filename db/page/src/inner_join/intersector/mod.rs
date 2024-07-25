use afs_primitives::is_less_than_tuple::IsLessThanTupleAir;

use self::columns::{IntersectorAuxCols, IntersectorCols, IntersectorIoCols};

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[derive(derive_new::new)]
pub struct IntersectorBuses {
    pub t1_intersector_bus_index: usize,
    pub t2_intersector_bus_index: usize,
    pub intersector_t2_bus_index: usize,
}

pub struct IntersectorAir {
    pub buses: IntersectorBuses,

    pub idx_len: usize,

    pub lt_chip: IsLessThanTupleAir,
}

impl IntersectorAir {
    pub fn new(
        range_bus_index: usize,
        buses: IntersectorBuses,
        idx_len: usize,
        idx_limb_bits: usize,
        decomp: usize,
    ) -> Self {
        Self {
            buses,
            idx_len,
            lt_chip: IsLessThanTupleAir::new(range_bus_index, vec![idx_limb_bits; idx_len], decomp),
        }
    }

    pub fn io_width(&self) -> usize {
        IntersectorIoCols::<usize>::width(self)
    }

    pub fn aux_width(&self) -> usize {
        IntersectorAuxCols::<usize>::width(self)
    }

    pub fn air_width(&self) -> usize {
        IntersectorCols::<usize>::width(self)
    }
}
