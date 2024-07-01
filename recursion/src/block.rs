use p3_field::AbstractField;
use serde::Deserialize;
use serde_derive::Serialize;

use afs_derive::AlignedBorrow;

// TODO: move somewhere else
pub const D: usize = 4;

/// The smallest unit of memory that can be read and written to.
#[derive(
    AlignedBorrow, Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize,
)]
#[repr(C)]
pub struct Block<T>(pub [T; D]);


impl<T> From<[T; D]> for Block<T> {
    fn from(arr: [T; D]) -> Self {
        Self(arr)
    }
}

impl<T: AbstractField> From<T> for Block<T> {
    fn from(value: T) -> Self {
        Self([value, T::zero(), T::zero(), T::zero()])
    }
}

impl<T: Copy> From<&[T]> for Block<T> {
    fn from(slice: &[T]) -> Self {
        let arr: [T; D] = slice.try_into().unwrap();
        Self(arr)
    }
}
