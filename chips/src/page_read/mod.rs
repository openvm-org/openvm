pub mod air;
pub mod chip;
pub mod columns;

pub struct PageReadChip {
    bus_index: usize,

    page_width: usize,
    page_height: usize,
}

impl PageReadChip {
    pub fn new(bus_index: usize, page: Vec<Vec<u32>>) -> Self {
        assert!(!page.is_empty());

        Self {
            bus_index,
            page_width: page[0].len(),
            page_height: page.len(),
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
