pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

// A enum for the different table types and their bus indices
pub enum TableType {
    T1 {
        t1_intersector_bus_index: usize,
        t1_output_bus_index: usize,
    },
    T2 {
        fkey_start: usize,
        fkey_end: usize,

        t2_intersector_bus_index: usize,
        intersector_t2_bus_index: usize,
        t2_output_bus_index: usize,
    },
}

pub struct TableAir {
    idx_len: usize,
    data_len: usize,

    table_type: TableType,
}

impl TableAir {
    pub fn new(idx_len: usize, data_len: usize, table_type: TableType) -> Self {
        Self {
            idx_len,
            data_len,
            table_type,
        }
    }

    pub fn table_width(&self) -> usize {
        1 + self.idx_len + self.data_len
    }

    pub fn air_width(&self) -> usize {
        1 + self.table_width()
    }
}
