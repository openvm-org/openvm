use sha2::{compress256, compress512, digest::generic_array::GenericArray};

use crate::{Sha2BlockHasherConfig, Sha2MainChipConfig};

pub const SHA2_REGISTER_READS: usize = 3;
pub const SHA2_READ_SIZE: usize = 4;
pub const SHA2_WRITE_SIZE: usize = 4;

#[derive(Clone)]
pub struct Sha256Config;

#[derive(Clone)]
pub struct Sha512Config;

#[derive(Clone)]
pub struct Sha384Config;

pub trait Sha2Config: Sha2MainChipConfig + Sha2BlockHasherConfig {
    fn compress(state: &mut [u8], input: &[u8]);
}

impl Sha2Config for Sha256Config {
    fn compress(state: &mut [u8], input: &[u8]) {
        let state: &mut [u32; 8] = unsafe { &mut *(state.as_mut_ptr() as *mut [u32; 8]) };
        let input_array = GenericArray::from_slice(input);
        compress256(state, &[*input_array]);
    }
}

impl Sha2Config for Sha512Config {
    fn compress(state: &mut [u8], input: &[u8]) {
        let state: &mut [u64; 8] = unsafe { &mut *(state.as_mut_ptr() as *mut [u64; 8]) };
        let input_array = GenericArray::from_slice(input);
        compress512(state, &[*input_array]);
    }
}

impl Sha2Config for Sha384Config {
    fn compress(state: &mut [u8], input: &[u8]) {
        let state: &mut [u64; 8] = unsafe { &mut *(state.as_mut_ptr() as *mut [u64; 8]) };
        let input_array = GenericArray::from_slice(input);
        compress512(state, &[*input_array]);
    }
}
