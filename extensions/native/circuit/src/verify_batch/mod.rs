//use crate::NATIVE_POSEIDON2_CHUNK_SIZE;

mod air;
pub mod chip;
mod columns;
#[cfg(test)]
mod tests;
mod trace;

const CHUNK: usize = 8; //NATIVE_POSEIDON2_CHUNK_SIZE;
