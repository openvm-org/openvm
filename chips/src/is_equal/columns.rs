use afs_derive::AlignedBorrow;

pub const NUM_COLS: usize = 4;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct IsEqualCols<T> {
    pub io: IsEqualIoCols<T>,
    pub aux: IsEqualAuxCols<T>,
}

#[derive(Clone, Copy)]
pub struct IsEqualIoCols<T> {
    pub x: T,
    pub y: T,
    pub is_equal: T,
}

#[derive(Debug, Clone)]
pub struct IsEqualAuxCols<T> {
    pub inv: T,
}

impl<T: Clone> IsEqualAuxCols<T> {
    pub fn from_slice(slc: &[T]) -> IsEqualAuxCols<T> {
        IsEqualAuxCols {
            inv: slc[0].clone(),
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        vec![self.inv.clone()]
    }
}
impl<T: Clone> IsEqualCols<T> {
    pub const fn new(x: T, y: T, is_equal: T, inv: T) -> IsEqualCols<T> {
        IsEqualCols {
            io: IsEqualIoCols { x, y, is_equal },
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

    pub fn flatten(&self) -> Vec<T> {
        vec![
            self.io.x.clone(),
            self.io.y.clone(),
            self.io.is_equal.clone(),
            self.aux.inv.clone(),
        ]
    }

    pub fn get_width() -> usize {
        NUM_COLS
    }
}
