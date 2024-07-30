use core::mem::size_of;

use afs_derive::AlignedBorrow;
use p3_keccak_air::KeccakCols;

#[repr(C)]
#[derive(Debug, AlignedBorrow)]
pub struct KeccakPermuteCols<T> {
    pub inner: KeccakCols<T>,
    pub io: KeccakPermuteIoCols<T>,
    pub aux: KeccakPermuteAuxCols<T>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, AlignedBorrow, derive_new::new)]
pub struct KeccakPermuteIoCols<T> {
    /// Whether row corresponds to an opcode (PERMUTE)
    pub is_opcode: T,
    /// The timestamp when the opcode was received.
    /// The execution of the opcode will use multiple timestamps, with `clk` as the initial one.
    pub clk: T,
    pub a: T,
    // pub b: T, // b = offset = 0
    pub c: T,
    pub d: T,
    pub e: T,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, AlignedBorrow, derive_new::new)]
pub struct KeccakPermuteAuxCols<T> {
    pub dst: T,
    pub src: T,
}

pub const NUM_KECCAK_PERMUTE_COLS: usize = size_of::<KeccakPermuteCols<u8>>();
