pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

pub struct IntersectorAir {
    pub t1_intersector_bus_index: usize,
    pub t2_intersector_bus_index: usize,
    pub intersector_t2_bus_index: usize,

    pub idx_len: usize,
}

impl IntersectorAir {
    pub fn new(
        t1_intersector_bus_index: usize,
        t2_intersector_bus_index: usize,
        intersector_t2_bus_index: usize,
        idx_len: usize,
    ) -> Self {
        Self {
            t1_intersector_bus_index,
            t2_intersector_bus_index,
            intersector_t2_bus_index,
            idx_len,
        }
    }

    pub fn air_width(&self) -> usize {
        self.idx_len + 4
    }
}
