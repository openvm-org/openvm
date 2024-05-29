#[cfg(test)]
pub mod tests;

pub mod air;
pub mod columns;
pub mod trace;

#[derive(Default)]
/// A chip that checks if a number equals 0
pub struct IsZeroChip {
    bus_index: usize,
    pub x: Vec<u32>,
}

impl IsZeroChip {
    pub fn new(bus_index: usize, x: Vec<u32>) -> Self {
        Self { bus_index, x }
    }

    pub fn bus_index(&self) -> usize {
        self.bus_index
    }

    fn is_zero(&self, x: u32) -> u32 {
        if x == 0 {
            1
        } else {
            0
        }
    }

    pub fn request(&self, x: u32) -> u32 {
        self.is_zero(x)
    }
}
