#[cfg(test)]
pub mod tests;

pub mod air;
pub mod columns;
pub mod trace;

#[derive(Default)]
pub struct IsEqualChip {
    pub x: Vec<u32>,
    pub y: Vec<u32>,
}

impl IsEqualChip {
    pub fn new(x: Vec<u32>, y: Vec<u32>) -> Self {
        Self { x, y }
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
