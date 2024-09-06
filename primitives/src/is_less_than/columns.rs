use afs_derive::AlignedBorrow;
use derive_new::new;
use p3_air::AirBuilder;

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
}

impl<T> IsLessThanIoCols<T> {
    pub fn flatten(self) -> Vec<T> {
        vec![self.x, self.y, self.less_than]
    }

    pub fn width() -> usize {
        3
    }
}

#[derive(Debug, Clone, PartialEq, Eq, new)]
pub struct IsLessThanAuxCols<T> {
    // lower_decomp consists of lower decomposed into limbs of size decomp where we also shift
    // the final limb and store it as the last element of lower decomp so we can range check
    pub lower_decomp: Vec<T>,
}

impl<T: Clone> IsLessThanAuxCols<T> {
    pub fn from_slice(slc: &[T]) -> Self {
        Self {
            lower_decomp: slc.to_vec(),
        }
    }
}

impl<T> IsLessThanAuxCols<T> {
    pub fn flatten(self) -> Vec<T> {
        self.lower_decomp
    }

    pub fn from_iterator<I: Iterator<Item = T>>(iter: &mut I, lt_air: &IsLessThanAir) -> Self {
        Self {
            lower_decomp: (0..Self::width(lt_air))
                .map(|_| iter.next().unwrap())
                .collect(),
        }
    }

    pub fn width(lt_air: &IsLessThanAir) -> usize {
        lt_air.num_limbs + (lt_air.max_bits % lt_air.decomp != 0) as usize
    }

    pub fn into_expr<AB: AirBuilder>(self) -> IsLessThanAuxCols<AB::Expr>
    where
        T: Into<AB::Expr>,
    {
        IsLessThanAuxCols::new(self.lower_decomp.into_iter().map(|x| x.into()).collect())
    }
}

#[derive(Clone, new)]
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
}

impl<T> IsLessThanCols<T> {
    pub fn flatten(self) -> Vec<T> {
        let mut flattened = self.io.flatten();
        flattened.extend(self.aux.flatten());
        flattened
    }

    pub fn width(lt_air: &IsLessThanAir) -> usize {
        IsLessThanIoCols::<T>::width() + IsLessThanAuxCols::<T>::width(lt_air)
    }
}

impl<T> IsLessThanIoCols<T> {
    pub fn new(x: impl Into<T>, y: impl Into<T>, less_than: impl Into<T>) -> Self {
        Self {
            x: x.into(),
            y: y.into(),
            less_than: less_than.into(),
        }
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
    pub lower_decomp: &'a mut [T],
}

impl<'a, T> IsLessThanAuxColsMut<'a, T> {
    pub fn from_slice(slc: &'a mut [T]) -> Self {
        Self { lower_decomp: slc }
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
