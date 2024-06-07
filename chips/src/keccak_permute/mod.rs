mod air;
mod chip;
mod columns;
mod trace;

#[derive(Clone)]
pub struct KeccakPermuteChip {
    pub bus_input: usize,
    pub bus_output: usize,
}
