pub mod air;
pub mod bridge;

<<<<<<< HEAD:chips/src/page_rw_checker/my_initial_page/mod.rs
#[derive(Debug, Clone)]
pub struct MyInitialPageAir {
=======
#[derive(Clone, Debug)]
pub struct PageReadAir {
>>>>>>> d74b0541394676b6966e07196adf50328a41d65b:chips/src/page_rw_checker/initial_page/mod.rs
    page_bus: usize,
    idx_len: usize,
    data_len: usize,
}

impl PageReadAir {
    pub fn new(page_bus: usize, idx_len: usize, data_len: usize) -> Self {
        Self {
            page_bus,
            idx_len,
            data_len,
        }
    }

    pub fn air_width(&self) -> usize {
        1 + self.idx_len + self.data_len
    }
}
