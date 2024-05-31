use afs_derive::AlignedBorrow;

pub const NUM_COLS: usize = 4;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct IsEqualCols<F> {
    pub io: IsEqualIOCols<F>,
    pub inv: F,
}

#[derive(Clone, Copy)]
pub struct IsEqualIOCols<T> {
    pub x: T,
    pub y: T,
    pub is_equal: T,
}

impl<T: Clone> IsEqualCols<T> {
    pub const fn new(x: T, y: T, is_equal: T, inv: T) -> IsEqualCols<T> {
        IsEqualCols {
            io: IsEqualIOCols { x, y, is_equal },
            inv,
        }
    }

    pub fn get_width() -> usize {
        NUM_COLS
    }
}
