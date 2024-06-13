pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

// A enum for the different table types and their bus indices
enum TableType {
    T1 {
        t1_intersector: usize,
        t1_output: usize,
    },
    T2 {
        fkey_start: usize,
        fkey_end: usize,

        t2_intersector: usize,
        intersector_t2: usize,
        t2_output: usize,
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
