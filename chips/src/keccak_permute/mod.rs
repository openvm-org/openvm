mod air;
mod chip;
mod columns;
mod constants;
mod logic;
mod round_flags;
mod trace;

pub const NUM_U64_HASH_ELEMS: usize = 4;
pub const NUM_ROUNDS: usize = 24;
const BITS_PER_LIMB: usize = 16;
pub const U64_LIMBS: usize = 64 / BITS_PER_LIMB;
const RATE_BITS: usize = 1088;
const RATE_LIMBS: usize = RATE_BITS / BITS_PER_LIMB;

#[derive(Clone)]
pub struct KeccakPermuteChip {
    pub bus_input: usize,
    pub bus_output: usize,
    pub bus_output_digest: usize,

    pub inputs_sponge: Vec<[u64; 25]>,
    pub inputs_digest: Vec<[u64; 25]>,
}
