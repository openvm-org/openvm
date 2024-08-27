use afs_derive::AlignedBorrow;
use derive_new::new;

pub const NUM_COLS: usize = 3;

#[repr(C)]
#[derive(AlignedBorrow, new)]
pub struct IsZeroCols<T> {
    pub io: IsZeroIoCols<T>,
    pub inv: T,
}

#[derive(Copy, Clone, new)]
pub struct IsZeroIoCols<F> {
    pub x: F,
    pub is_zero: F,
}

impl<F: Clone> IsZeroCols<F> {
    pub fn flatten(&self) -> Vec<F> {
        vec![self.io.x.clone(), self.io.is_zero.clone(), self.inv.clone()]
    }

    pub fn get_width() -> usize {
        NUM_COLS
    }
}
