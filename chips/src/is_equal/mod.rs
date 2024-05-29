#[cfg(test)]
pub mod tests;

pub mod air;
pub mod columns;
pub mod trace;

pub struct IsEqualChip {
    bus_index: usize,
    pub x: Vec<u32>,
    pub y: Vec<u32>,
}

impl IsEqualChip {
    pub fn new(bus_index: usize, x: Vec<u32>, y: Vec<u32>) -> Self {
        Self { bus_index, x, y }
    }

    pub fn bus_index(&self) -> usize {
        self.bus_index
    }

    fn is_equal(&self, x: u32, y: u32) -> u32 {
        if x == y {
            1
        } else {
            0
        }
    }

    pub fn request(&self, x: u32, y: u32) -> u32 {
        self.is_equal(x, y)
    }
}
