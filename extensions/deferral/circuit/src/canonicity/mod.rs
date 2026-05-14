use openvm_circuit_primitives::{StructReflection, StructReflectionHelper};
use openvm_circuit_primitives_derive::AlignedBorrow;

use crate::utils::F_NUM_U16S;

mod air;
mod trace;

pub use air::*;
pub use trace::*;

/// Number of limbs the canonicity sub-AIR walks over. Set to `F_NUM_U16S` (2)
/// after the u16 cell-shape migration. The sub-AIR is generic over this constant
/// (and the corresponding limb bit-width) so the same proof template covers any
/// limb decomposition of an F element.
pub const CANONICITY_NUM_LIMBS: usize = F_NUM_U16S;
/// Bit width of each limb the canonicity sub-AIR walks over.
pub const CANONICITY_LIMB_BITS: usize = 16;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CanonicityIo<T> {
    pub x: [T; CANONICITY_NUM_LIMBS],
    /// Assumed boolean by caller.
    pub count: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Clone, Copy, Debug)]
pub struct CanonicityAuxCols<T> {
    /// Marker for the first index where x[i] != order[i] (big-endian).
    pub diff_marker: [T; CANONICITY_NUM_LIMBS],
    /// order[i] - x[i] at the first differing index. Bounded by
    /// `2^CANONICITY_LIMB_BITS - 1` because each limb is range-checked.
    pub diff_val: T,
}
