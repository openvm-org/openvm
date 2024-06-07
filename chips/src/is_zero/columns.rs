use afs_derive::AlignedBorrow;

pub const NUM_COLS: usize = 3;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct IsZeroCols<T> {
    pub io: IsZeroIOCols<T>,
    pub inv: T,
}

#[derive(Copy, Clone)]
pub struct IsZeroIOCols<T> {
    pub x: T,
    pub is_zero: T,
}

impl<T> IsZeroCols<T> {
    pub const fn new(x: T, is_zero: T, inv: T) -> IsZeroCols<T> {
        IsZeroCols {
            io: IsZeroIOCols { x, is_zero },
            inv,
        }
    }

    pub fn flatten(&self) -> Vec<T>
    where
        T: Clone,
    {
        vec![self.io.x.clone(), self.io.is_zero.clone(), self.inv.clone()]
    }

    pub fn get_width() -> usize {
        NUM_COLS
    }
}
