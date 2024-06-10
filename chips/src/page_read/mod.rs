pub mod air;
pub mod bridge;
pub mod columns;
pub mod page_controller;

pub struct PageReadAir {
    bus_index: usize,

    page_width: usize,
    page_height: usize,
}

impl PageReadAir {
    pub fn new(bus_index: usize, page_width: usize, page_height: usize) -> Self {
        Self {
            bus_index,
            page_width,
            page_height,
        }
    }

    pub fn bus_index(&self) -> usize {
        self.bus_index
    }

    pub fn page_height(&self) -> usize {
        self.page_height
    }

    pub fn page_width(&self) -> usize {
        self.page_width
    }

    pub fn air_width(&self) -> usize {
        2 + self.page_width
    }
}
