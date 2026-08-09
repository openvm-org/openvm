//! Sub-AIR constraining that a little-endian byte decomposition of a field element is *canonical*,
//! i.e. that the composed integer is `< F::ORDER_U32`.
//!
//! Byte-range checks alone do not pin down a field element: for any `y < p` the byte strings
//! encoding `y` and `y + p` compose to the same field element whenever `y + p < 2^32` (always true
//! for BabyBear, where `p < 2^31`). Any chip that lets the prover choose the bytes — e.g. bytes
//! written back to memory — must additionally constrain canonicity.
//!
//! NOTE: `openvm-deferral-circuit` carries an identical sub-AIR in its own `canonicity` module.
//! Both should eventually be hoisted into `openvm-circuit-primitives`; that is deliberately left
//! out of this PR to keep the diff inside the poseidon2 extension.

use openvm_circuit_primitives::{StructReflection, StructReflectionHelper};
use openvm_circuit_primitives_derive::AlignedBorrow;

mod air;
mod trace;

pub use air::*;
pub use trace::*;

/// Number of bytes in the little-endian representation of `F::ORDER_U32`.
pub const F_NUM_BYTES: usize = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CanonicityIo<T> {
    pub x: [T; F_NUM_BYTES],
    /// Assumed boolean by caller.
    pub count: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Clone, Copy, Debug)]
pub struct CanonicityAuxCols<T> {
    /// Marker for the first index where `x[i] != order[i]` (big-endian).
    pub diff_marker: [T; F_NUM_BYTES],
    /// `order[i] - x[i]` at the first differing index, constrained to `[1, 255]`.
    pub diff_val: T,
}
