use afs_derive::AlignedBorrow;

pub const NUM_COLS: usize = 4;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct IsEqualCols<T> {
    pub io: IsEqualIOCols<T>,
    pub aux: IsEqualAuxCols<T>,
}

#[derive(Clone, Copy)]
pub struct IsEqualIOCols<T> {
    pub x: T,
    pub y: T,
    pub is_equal: T,
}

pub struct IsEqualAuxCols<T> {
    pub inv: T,
}

impl<T: Clone> IsEqualCols<T> {
    pub const fn new(x: T, y: T, is_equal: T, inv: T) -> IsEqualCols<T> {
        IsEqualCols {
            io: IsEqualIOCols { x, y, is_equal },
            aux: IsEqualAuxCols { inv },
        }
    }

    pub fn from_slice(slc: &[T]) -> IsEqualCols<T> {
        let x = slc[0].clone();
        let y = slc[1].clone();
        let is_equal = slc[2].clone();
        let inv = slc[3].clone();

        IsEqualCols::new(x, y, is_equal, inv)
    }

    pub fn get_width() -> usize {
        NUM_COLS
    }
}
