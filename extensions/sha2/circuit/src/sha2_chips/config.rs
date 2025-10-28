use itertools::Itertools;
use openvm_sha2_air::{Sha256Config, Sha2BlockHasherSubairConfig, Sha384Config, Sha512Config};
use sha2::{
    compress256, compress512, digest::generic_array::GenericArray, Digest, Sha256, Sha384, Sha512,
};

use crate::{Sha2BlockHasherVmConfig, Sha2MainChipConfig};

pub const SHA2_REGISTER_READS: usize = 3;
pub const SHA2_READ_SIZE: usize = 4;
pub const SHA2_WRITE_SIZE: usize = 4;

pub trait Sha2Config: Sha2MainChipConfig + Sha2BlockHasherVmConfig {
    /// Number of bits used to store the message length (part of the message padding)
    const MESSAGE_LENGTH_BITS: usize;

    // Preconditions:
    // - state.len() >= Self::STATE_BYTES
    // - input.len() == Self::BLOCK_BYTES
    fn compress(state: &mut [u8], input: &[u8]);

    fn hash(message: &[u8]) -> Vec<u8>;
}

impl Sha2Config for Sha256Config {
    const MESSAGE_LENGTH_BITS: usize = 64;

    // TODO: do this faster
    fn compress(state: &mut [u8], input: &[u8]) {
        debug_assert!(state.len() >= Sha256Config::STATE_BYTES);
        debug_assert!(input.len() == Sha256Config::BLOCK_BYTES);

        let state_u32s = state
            .chunks_exact(4)
            .map(|chunk| u32::from_be_bytes(chunk.try_into().unwrap()))
            .collect_vec();
        let mut state_u32s_array = state_u32s.try_into().unwrap();
        // let state: &mut [u32; 8] = unsafe { &mut *(state.as_mut_ptr() as *mut [u32; 8]) };
        let input_array = GenericArray::from_slice(input);
        compress256(&mut state_u32s_array, &[*input_array]);

        state.copy_from_slice(
            &state_u32s_array
                .iter()
                .flat_map(|x| x.to_be_bytes())
                .collect_vec(),
        );
    }

    fn hash(message: &[u8]) -> Vec<u8> {
        Sha256::digest(message).to_vec()
    }
}

impl Sha2Config for Sha512Config {
    const MESSAGE_LENGTH_BITS: usize = 128;

    fn compress(state: &mut [u8], input: &[u8]) {
        debug_assert!(state.len() >= Sha512Config::STATE_BYTES);
        debug_assert!(input.len() == Sha512Config::BLOCK_BYTES);
        let state: &mut [u64; 8] = unsafe { &mut *(state.as_mut_ptr() as *mut [u64; 8]) };
        let input_array = GenericArray::from_slice(input);
        compress512(state, &[*input_array]);
    }

    fn hash(message: &[u8]) -> Vec<u8> {
        Sha512::digest(message).to_vec()
    }
}

impl Sha2Config for Sha384Config {
    const MESSAGE_LENGTH_BITS: usize = Sha512Config::MESSAGE_LENGTH_BITS;

    fn compress(state: &mut [u8], input: &[u8]) {
        Sha512Config::compress(state, input);
    }

    fn hash(message: &[u8]) -> Vec<u8> {
        Sha384::digest(message).to_vec()
    }
}
