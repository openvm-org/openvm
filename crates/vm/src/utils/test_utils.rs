use std::array;

use openvm_circuit::arch::{MemoryConfig, SystemConfig};
use openvm_instructions::{
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
    NATIVE_AS,
};
use openvm_stark_backend::p3_field::PrimeField32;
use rand::{rngs::StdRng, Rng};

use crate::system::memory::{merkle::public_values::PUBLIC_VALUES_AS, online::PAGE_SIZE};

pub fn i32_to_f<F: PrimeField32>(val: i32) -> F {
    if val.signum() == -1 {
        -F::from_canonical_u32(val.unsigned_abs())
    } else {
        F::from_canonical_u32(val as u32)
    }
}

pub fn generate_long_number<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    rng: &mut StdRng,
) -> [u32; NUM_LIMBS] {
    array::from_fn(|_| rng.gen_range(0..(1 << LIMB_BITS)))
}

// in little endian
pub fn u32_into_limbs<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    num: u32,
) -> [u32; NUM_LIMBS] {
    array::from_fn(|i| (num >> (LIMB_BITS * i)) & ((1 << LIMB_BITS) - 1))
}

pub fn u32_sign_extend<const IMM_BITS: usize>(num: u32) -> u32 {
    if num & (1 << (IMM_BITS - 1)) != 0 {
        num | (u32::MAX - (1 << IMM_BITS) + 1)
    } else {
        num
    }
}

pub fn test_system_config_without_continuations() -> SystemConfig {
    let mut addr_spaces = MemoryConfig::empty_address_space_configs(5);
    addr_spaces[RV32_REGISTER_AS as usize].num_cells = PAGE_SIZE;
    addr_spaces[RV32_MEMORY_AS as usize].num_cells = 1 << 22;
    addr_spaces[PUBLIC_VALUES_AS as usize].num_cells = PAGE_SIZE;
    addr_spaces[NATIVE_AS as usize].num_cells = 1 << 25;
    SystemConfig::new(3, MemoryConfig::new(2, addr_spaces, 29, 29, 17, 32), 32)
        .without_continuations()
}

// Testing config when native address space is not needed, with continuations enabled
pub fn test_system_config() -> SystemConfig {
    let mut config = test_system_config_without_continuations();
    config.memory_config.addr_spaces[NATIVE_AS as usize].num_cells = 0;
    config.with_continuations()
}

/// Generate a random message of a given length in bytes
pub fn get_random_message(rng: &mut StdRng, len: usize) -> Vec<u8> {
    let mut random_message: Vec<u8> = vec![0u8; len];
    rng.fill(&mut random_message[..]);
    random_message
}
