use openvm_circuit_primitives::AlignedBorrow;
use openvm_poseidon2_air::Poseidon2SubCols;

/// Columns for Poseidon2Vm AIR.
#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Poseidon2PeripheryCols<
    F,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const PARTIAL_ROUNDS: usize,
> {
    pub inner: Poseidon2SubCols<F, SBOX_DEGREE, SBOX_REGISTERS, PARTIAL_ROUNDS>,
    pub mult: F,
}
