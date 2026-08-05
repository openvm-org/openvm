//! Append-only public outputs committed independently from guest memory.

use openvm_instructions::VM_DIGEST_WIDTH;
use openvm_stark_backend::p3_field::{PrimeCharacteristicRing, PrimeField32};

use crate::arch::{hasher::Hasher, PublicValuesState, U16_CELLS_PER_PUBLIC_VALUE, U16_CELL_SIZE};

mod air;
mod bus;
pub mod proof;
mod trace;

#[cfg(test)]
mod tests;

pub use air::*;
pub use bus::*;
pub use trace::*;

/// Number of little-endian `u16` limbs in one public `u64` value.
pub const PUBLIC_VALUE_LIMBS: usize = U16_CELLS_PER_PUBLIC_VALUE;

/// Domain separator stored in the initial public-output accumulator.
pub const PUBLIC_VALUES_INIT_DOMAIN: u32 = 0x5056_1001;
/// Domain separator absorbed for every appended public output.
pub const PUBLIC_VALUES_EVENT_DOMAIN: u32 = 0x5056_1002;

/// The public-output AIR has one row per configured output slot.
pub const fn public_values_trace_height(num_public_value_cells: usize) -> usize {
    num_public_value_cells / PUBLIC_VALUE_LIMBS
}

/// Maximum number of Poseidon2 records the AIR can emit in one segment.
pub const fn public_values_poseidon2_record_count(num_public_value_cells: usize) -> usize {
    public_values_trace_height(num_public_value_cells)
}

/// Initial accumulator for an empty output stream with the configured capacity.
pub fn public_values_initial_commit<F: PrimeCharacteristicRing>(
    max_values: usize,
) -> [F; VM_DIGEST_WIDTH] {
    let mut commit = [F::ZERO; VM_DIGEST_WIDTH];
    commit[0] = F::from_u32(PUBLIC_VALUES_INIT_DOMAIN);
    commit[1] = F::from_usize(max_values);
    commit
}

/// Right compression input for one output event.
pub fn public_values_event_block<F: PrimeField32>(value: u64) -> [F; VM_DIGEST_WIDTH] {
    public_values_event_block_from_limbs(value_limbs(value))
}

/// Right compression input for one output event already encoded as `u16` limbs.
pub fn public_values_event_block_from_limbs<F: PrimeCharacteristicRing>(
    value: [F; PUBLIC_VALUE_LIMBS],
) -> [F; VM_DIGEST_WIDTH] {
    let mut block = [F::ZERO; VM_DIGEST_WIDTH];
    block[0] = F::from_u32(PUBLIC_VALUES_EVENT_DOMAIN);
    for (dst, src) in block[1..1 + PUBLIC_VALUE_LIMBS].iter_mut().zip(value) {
        *dst = src;
    }
    block
}

/// Commits an ordered public-output prefix.
pub fn public_values_commit<F: PrimeField32>(
    values: &[u64],
    max_values: usize,
    hasher: &impl Hasher<VM_DIGEST_WIDTH, F>,
) -> [F; VM_DIGEST_WIDTH] {
    assert!(values.len() <= max_values);
    values.iter().fold(
        public_values_initial_commit(max_values),
        |commit, &value| hasher.compress(&commit, &public_values_event_block(value)),
    )
}

/// Encodes every configured slot as four `u16` cells, padding the unpublished suffix with zero.
pub(crate) fn public_values_cells<F: PrimeField32>(state: &PublicValuesState) -> Vec<F> {
    state
        .values()
        .iter()
        .copied()
        .chain(std::iter::repeat_n(
            0,
            state.max_public_values().saturating_sub(state.len()),
        ))
        .flat_map(value_limbs)
        .collect()
}

pub(crate) fn value_limbs<F: PrimeField32>(value: u64) -> [F; PUBLIC_VALUE_LIMBS] {
    [
        F::from_u16(value as u16),
        F::from_u16((value >> 16) as u16),
        F::from_u16((value >> 32) as u16),
        F::from_u16((value >> 48) as u16),
    ]
}

pub const fn public_values_cells_from_bytes(num_public_values_bytes: usize) -> usize {
    assert!(
        num_public_values_bytes.is_multiple_of(U16_CELL_SIZE),
        "num_public_values_bytes must be a multiple of the u16 cell size"
    );
    num_public_values_bytes / U16_CELL_SIZE
}

pub const fn assert_public_values_shape(num_cells: usize) {
    assert!(
        num_cells.is_multiple_of(PUBLIC_VALUE_LIMBS),
        "public values must contain complete u64 outputs"
    );
    assert!(
        (num_cells / PUBLIC_VALUE_LIMBS).is_power_of_two(),
        "public-values capacity must be a power of two"
    );
}
