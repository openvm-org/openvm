use afs_derive::AlignedBorrow;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct IsLessThanBitsIOCols<F> {
    pub x: F,
    pub y: F,
    pub is_less_than: F,
}

#[derive(Clone)]
pub struct IsLessThanBitsAuxCols<F> {
    pub source_bits: Vec<F>,
}

pub struct IsLessThanBitsCols<F> {
    pub io: IsLessThanBitsIOCols<F>,
    pub aux: IsLessThanBitsAuxCols<F>,
}

// copied from is_less_than
impl<T: Clone> IsLessThanBitsIOCols<T> {
    pub fn from_slice(slc: &[T]) -> Self {
        Self {
            x: slc[0].clone(),
            y: slc[1].clone(),
            is_less_than: slc[2].clone(),
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        vec![self.x.clone(), self.y.clone(), self.is_less_than.clone()]
    }

    pub fn get_width() -> usize {
        3
    }
}

impl<T: Clone> IsLessThanBitsAuxCols<T> {
    pub fn from_slice(slc: &[T]) -> Self {
        Self {
            source_bits: slc.to_vec(),
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![];
        flattened.extend(self.source_bits.iter().cloned());
        flattened
    }

    pub fn get_width(limb_bits: usize) -> usize {
        limb_bits + 1
    }
}

impl<T: Clone> IsLessThanBitsCols<T> {
    pub fn from_slice(slc: &[T]) -> Self {
        let io = IsLessThanBitsIOCols::from_slice(&slc[..3]);
        let aux = IsLessThanBitsAuxCols::from_slice(&slc[3..]);

        Self { io, aux }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = self.io.flatten();
        flattened.extend(self.aux.flatten());
        flattened
    }

    pub fn get_width(limb_bits: usize) -> usize {
        IsLessThanBitsIOCols::<T>::get_width() + IsLessThanBitsAuxCols::<T>::get_width(limb_bits)
    }
}
