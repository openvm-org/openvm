/// Stateful keccak256 hasher. Handles full keccak sponge (padding, absorb, keccak-f) on
/// variable length inputs read from VM memory.
pub mod hasher;
