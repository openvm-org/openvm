use afs_derive::AlignedBorrow;

#[derive(Default, AlignedBorrow)]
pub struct IsLessThanIOCols<T> {
    pub x: T,
    pub y: T,
    pub less_than: T,
}

impl<T: Clone> IsLessThanIOCols<T> {
    pub fn from_slice(slc: &[T]) -> Self {
        Self {
            x: slc[0].clone(),
            y: slc[1].clone(),
            less_than: slc[2].clone(),
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        vec![self.x.clone(), self.y.clone(), self.less_than.clone()]
    }

    pub fn get_width() -> usize {
        3
    }
}

pub struct IsLessThanAuxCols<T> {
    pub lower: T,
    // lower_decomp consists of lower decomposed into limbs of size decomp where we also shift
    // the final limb and store it as the last element of lower decomp so we can range check
    pub lower_decomp: Vec<T>,
}

impl<T: Clone> IsLessThanAuxCols<T> {
    pub fn from_slice(slc: &[T]) -> Self {
        Self {
            lower: slc[0].clone(),
            lower_decomp: slc[1..].to_vec(),
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![self.lower.clone()];
        flattened.extend(self.lower_decomp.iter().cloned());
        flattened
    }

    pub fn get_width(limb_bits: usize, decomp: usize) -> usize {
        let mut width = 0;
        // for the lower
        width += 1;
        // for the decomposed lower
        let num_limbs = (limb_bits + decomp - 1) / decomp;
        width += num_limbs + 1;

        width
    }
}

pub struct IsLessThanCols<T> {
    pub io: IsLessThanIOCols<T>,
    pub aux: IsLessThanAuxCols<T>,
}

impl<T: Clone> IsLessThanCols<T> {
    pub fn from_slice(slc: &[T]) -> Self {
        let io = IsLessThanIOCols::from_slice(&slc[..3]);
        let aux = IsLessThanAuxCols::from_slice(&slc[3..]);

        Self { io, aux }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = self.io.flatten();
        flattened.extend(self.aux.flatten());
        flattened
    }

    pub fn get_width(limb_bits: usize, decomp: usize) -> usize {
        IsLessThanIOCols::<T>::get_width() + IsLessThanAuxCols::<T>::get_width(limb_bits, decomp)
    }
}
