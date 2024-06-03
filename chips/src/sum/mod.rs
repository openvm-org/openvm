pub mod air;
pub mod columns;
pub mod trace;

pub struct SumChip {
    pub bus_input: usize,
}

impl SumChip {
    pub fn new(bus_input: usize) -> Self {
        Self { bus_input }
    }
}
