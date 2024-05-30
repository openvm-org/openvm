use afs_derive::AlignedBorrow;

pub const NUM_COLS: usize = 3;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct IsZeroCols<F> {
    pub x: F,
    pub is_zero: F,
    pub inv: F,
}

impl<F: Clone> IsZeroCols<F> {
    pub const fn new(x: F, is_zero: F, inv: F) -> IsZeroCols<F> {
        IsZeroCols { x, is_zero, inv }
    }

    pub fn from_slice(slc: &[F]) -> IsZeroCols<F> {
        IsZeroCols {
            x: slc[0].clone(),
            is_zero: slc[1].clone(),
            inv: slc[2].clone(),
        }
    }

    pub fn get_width() -> usize {
        NUM_COLS
    }
}
