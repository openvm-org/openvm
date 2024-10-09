mod rv32_alu;
mod rv32_heap;
mod rv32_loadstore;
mod rv32_mul;

pub use rv32_alu::*;
pub use rv32_heap::*;
pub use rv32_loadstore::*;
pub use rv32_mul::*;

/// 32-bit register stored as 4 bytes (4 lanes of 8-bits)
pub const RV32_REGISTER_NUM_LANES: usize = 4;

// For soundness, should be <= 16
pub const RV_IS_TYPE_IMM_BITS: usize = 12;

use p3_field::PrimeField32;
/// Convert the RISC-V register data (32 bits represented as 4 bytes, where each byte is represented as a field element)
/// back into its value as u32.
pub fn compose<F: PrimeField32>(ptr_data: [F; 4]) -> u32 {
    let mut val = 0;
    for (i, limb) in ptr_data.map(|x| x.as_canonical_u32()).iter().enumerate() {
        val += limb << (i * 8);
    }
    val
}
