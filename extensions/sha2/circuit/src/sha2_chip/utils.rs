use crate::Sha2ChipConfig;

/// Returns the number of blocks required to hash a message of length `len`
pub fn get_sha2_num_blocks<C: Sha2ChipConfig>(len: u32) -> u32 {
    // need to pad with one 1 bit, 64 bits for the message length and then pad until the length
    // is divisible by [C::BLOCK_BITS]
    ((len << 3) as usize + 1 + C::MESSAGE_LENGTH_BITS).div_ceil(C::BLOCK_BITS) as u32
}
