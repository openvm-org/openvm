mod air;
mod chip;
pub mod columns;
mod trace;
mod util;

#[derive(Default, Clone)]
pub struct KeccakSpongeOp {
    pub timestamp: u32,
    pub addr: u32,
    pub input: Vec<u8>,
}

#[derive(Default, Clone)]
pub struct KeccakSpongeAir {
    pub bus_input: usize,
    pub bus_output: usize,

    pub bus_xor_input: usize,
    pub bus_xor_output: usize,

    pub bus_permute_input: usize,
    pub bus_permute_output: usize,
}
