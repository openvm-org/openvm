use afs_derive::AlignedBorrow;

use super::IsLessThanAir;

#[derive(Default, AlignedBorrow, Clone)]
pub struct IsLessThanIoCols<T> {
    pub x: T,
    pub y: T,
    pub less_than: T,
}

impl<T: Clone> IsLessThanIoCols<T> {
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

    pub fn width() -> usize {
        3
    }
}

#[derive(Debug, Clone)]
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
}

impl<T> IsLessThanAuxCols<T> {
    pub fn width(lt_air: &IsLessThanAir) -> usize {
        1 + lt_air.num_limbs + (lt_air.max_bits % lt_air.decomp != 0) as usize
    }
}

pub struct IsLessThanCols<T> {
    pub io: IsLessThanIoCols<T>,
    pub aux: IsLessThanAuxCols<T>,
}

impl<T: Clone> IsLessThanCols<T> {
    pub fn from_slice(slc: &[T]) -> Self {
        let io = IsLessThanIoCols::from_slice(&slc[..3]);
        let aux = IsLessThanAuxCols::from_slice(&slc[3..]);

        Self { io, aux }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = self.io.flatten();
        flattened.extend(self.aux.flatten());
        flattened
    }

    pub fn width(lt_air: &IsLessThanAir) -> usize {
        IsLessThanIoCols::<T>::width() + IsLessThanAuxCols::<T>::width(lt_air)
    }
}

pub struct IsLessThanIoColsMut<'a, T> {
    pub x: &'a mut T,
    pub y: &'a mut T,
    pub less_than: &'a mut T,
}

impl<'a, T> IsLessThanIoColsMut<'a, T> {
    pub fn from_slice(slc: &'a mut [T]) -> Self {
        let (x, rest) = slc.split_first_mut().unwrap();
        let (y, rest) = rest.split_first_mut().unwrap();
        let (less_than, _) = rest.split_first_mut().unwrap();

        Self { x, y, less_than }
    }
}

pub struct IsLessThanAuxColsMut<'a, T> {
    pub lower: &'a mut T,
    pub lower_decomp: &'a mut [T],
}

impl<'a, T> IsLessThanAuxColsMut<'a, T> {
    pub fn from_slice(slc: &'a mut [T]) -> Self {
        let (lower, lower_decomp) = slc.split_first_mut().unwrap();

        Self {
            lower,
            lower_decomp,
        }
    }
}

pub struct IsLessThanColsMut<'a, T> {
    pub io: IsLessThanIoColsMut<'a, T>,
    pub aux: IsLessThanAuxColsMut<'a, T>,
}

impl<'a, T> IsLessThanColsMut<'a, T> {
    pub fn from_slice(slc: &'a mut [T]) -> Self {
        let (io, aux) = slc.split_at_mut(3);

        let io = IsLessThanIoColsMut::from_slice(io);
        let aux = IsLessThanAuxColsMut::from_slice(aux);

        Self { io, aux }
    }
}
