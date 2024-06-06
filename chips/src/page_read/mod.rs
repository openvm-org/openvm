pub mod air;
pub mod bridge;
pub mod columns;

pub struct PageReadAir {
    pub bus_index: usize,
    pub width: usize,
}

pub struct PageReadChip {
    pub air: PageReadAir,

    pub page_width: usize,
    pub page_height: usize,
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
}
