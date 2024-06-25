pub mod air;
pub mod bridge;
pub mod columns;
pub mod page_controller;

pub struct PageReadAir {
    bus_index: usize,
    width: usize,
}

impl PageReadAir {
    pub fn bus_index(&self) -> usize {
        self.bus_index
    }

    pub fn width(&self) -> usize {
        self.width
    }
}

pub struct PageReadChip {
    pub air: PageReadAir,

    page_width: usize,
    page_height: usize,
}

impl PageReadChip {
    pub fn new(bus_index: usize, page: Vec<Vec<u32>>) -> Self {
        assert!(!page.is_empty());

        let page_width = page[0].len();
        let page_height = page.len();

        Self {
            air: PageReadAir {
                bus_index,
                width: page_width + 2,
            },
            page_width,
            page_height,
        }
    }

    pub fn page_width(&self) -> usize {
        self.page_width
    }

    pub fn page_height(&self) -> usize {
        self.page_height
    }
}
