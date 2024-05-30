use afs_derive::AlignedBorrow;

pub const NUM_COLS: usize = 4;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct IsEqualCols<F> {
    pub x: F,
    pub y: F,
    pub is_equal: F,
    pub inv: F,
}

impl<F: Clone> IsEqualCols<F> {
    pub const fn new(x: F, y: F, is_equal: F, inv: F) -> IsEqualCols<F> {
        IsEqualCols {
            x,
            y,
            is_equal,
            inv,
        }
    }

    pub fn from_slice(slc: &[F]) -> IsEqualCols<F> {
        IsEqualCols {
            x: slc[0].clone(),
            y: slc[1].clone(),
            is_equal: slc[2].clone(),
            inv: slc[3].clone(),
        }
    }

    pub fn get_width() -> usize {
        NUM_COLS
    }
}
