use openvm_circuit_primitives::{StructReflection, StructReflectionHelper};
use openvm_circuit_primitives_derive::AlignedBorrow;

use crate::utils::F_NUM_BYTES;

mod air;
mod trace;

pub use air::*;
pub use trace::*;

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
    /// Marker for the first index where x[i] != order[i] (big-endian).
    pub diff_marker: [T; F_NUM_BYTES],
    /// order[i] - x[i] at the first differing index, constrained to [1, 255].
    pub diff_val: T,
}
