use afs_derive::AlignedBorrow;

pub const NUM_COLS: usize = 3;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct IsZeroCols<T> {
    pub io: IsZeroIoCols<T>,
    pub inv: T,
}

#[derive(Copy, Clone)]
pub struct IsZeroIoCols<F> {
    pub x: F,
    pub is_zero: F,
}

impl<F: Clone> IsZeroCols<F> {
    pub const fn new(x: F, is_zero: F, inv: F) -> IsZeroCols<F> {
        IsZeroCols {
            io: IsZeroIoCols { x, is_zero },
            inv,
        }
    }

    pub fn flatten(&self) -> Vec<F> {
        vec![self.io.x.clone(), self.io.is_zero.clone(), self.inv.clone()]
    }

    pub fn get_width() -> usize {
        NUM_COLS
    }
}
