use afs_derive::AlignedBorrow;
use derive_new::new;

#[repr(C)]
#[derive(AlignedBorrow, Clone, Copy, Debug, Default)]
pub struct AssertLessThanIoCols<T> {
    pub x: T,
    pub y: T,
}

impl<T> AssertLessThanIoCols<T> {
    pub fn new(x: impl Into<T>, y: impl Into<T>) -> Self {
        Self {
            x: x.into(),
            y: y.into(),
        }
    }
}

/// AUX_LEN is the number of AUX columns
/// we have that AUX_LEN = (max_bits + bus.range_max_bits - 1) / bus.range_max_bits
#[repr(C)]
#[derive(AlignedBorrow, Clone, Copy, Debug, Eq, new, PartialEq)]
pub struct AssertLessThanAuxCols<T, const AUX_LEN: usize> {
    // lower_decomp consists of lower decomposed into limbs of size bus.range_max_bits
    // note: the final limb might have less than bus.range_max_bits bits
    pub lower_decomp: [T; AUX_LEN],
}

// repr(C) is needed to make sure that the compiler does not reorder the fields
// we assume the order of the fields when using borrow or borrow_mut
#[repr(C)]
#[derive(AlignedBorrow, Clone, Copy, Debug, new)]
pub struct AssertLessThanCols<T, const AUX_LEN: usize> {
    pub io: AssertLessThanIoCols<T>,
    pub aux: AssertLessThanAuxCols<T, AUX_LEN>,
}
