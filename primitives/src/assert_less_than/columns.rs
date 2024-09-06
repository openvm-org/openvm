use afs_derive::AlignedBorrow;
use derive_new::new;
use p3_air::AirBuilder;

use super::AssertLessThanAir;

#[derive(Default, AlignedBorrow, Clone)]
pub struct AssertLessThanIoCols<T> {
    pub x: T,
    pub y: T,
}

impl<T: Clone> AssertLessThanIoCols<T> {
    pub fn from_slice(slc: &[T]) -> Self {
        Self {
            x: slc[0].clone(),
            y: slc[1].clone(),
        }
    }
}

impl<T> AssertLessThanIoCols<T> {
    pub fn flatten(self) -> Vec<T> {
        vec![self.x, self.y]
    }

    pub fn width() -> usize {
        2
    }
}

#[derive(Debug, Clone, PartialEq, Eq, new)]
pub struct AssertLessThanAuxCols<T> {
    // lower_decomp consists of lower decomposed into limbs of size decomp where we also shift
    // the final limb and store it as the last element of lower decomp so we can range check
    pub lower_decomp: Vec<T>,
}

impl<T: Clone> AssertLessThanAuxCols<T> {
    pub fn from_slice(slc: &[T]) -> Self {
        Self {
            lower_decomp: slc.to_vec(),
        }
    }
}

impl<T> AssertLessThanAuxCols<T> {
    pub fn flatten(self) -> Vec<T> {
        self.lower_decomp
    }

    pub fn try_from_iter<I: Iterator<Item = T>>(iter: &mut I, lt_air: &AssertLessThanAir) -> Self {
        Self {
            lower_decomp: (0..Self::width(lt_air))
                .map(|_| iter.next().unwrap())
                .collect(),
        }
    }

    pub fn width(lt_air: &AssertLessThanAir) -> usize {
        lt_air.num_limbs + (lt_air.max_bits % lt_air.decomp != 0) as usize 
    }

    pub fn into_expr<AB: AirBuilder>(self) -> AssertLessThanAuxCols<AB::Expr>
    where
        T: Into<AB::Expr>,
    {
        AssertLessThanAuxCols::new(self.lower_decomp.into_iter().map(|x| x.into()).collect())
    }
}

#[derive(Clone, new)]
pub struct AssertLessThanCols<T> {
    pub io: AssertLessThanIoCols<T>,
    pub aux: AssertLessThanAuxCols<T>,
}

impl<T: Clone> AssertLessThanCols<T> {
    pub fn from_slice(slc: &[T]) -> Self {
        let io = AssertLessThanIoCols::from_slice(&slc[..2]);
        let aux = AssertLessThanAuxCols::from_slice(&slc[2..]);

        Self { io, aux }
    }
}

impl<T> AssertLessThanCols<T> {
    pub fn flatten(self) -> Vec<T> {
        let mut flattened = self.io.flatten();
        flattened.extend(self.aux.flatten());
        flattened
    }

    pub fn width(lt_air: &AssertLessThanAir) -> usize {
        AssertLessThanIoCols::<T>::width() + AssertLessThanAuxCols::<T>::width(lt_air)
    }
}

impl<T> AssertLessThanIoCols<T> {
    pub fn new(x: impl Into<T>, y: impl Into<T>) -> Self {
        Self {
            x: x.into(),
            y: y.into(),
        }
    }
}

pub struct AssertLessThanIoColsMut<'a, T> {
    pub x: &'a mut T,
    pub y: &'a mut T,
}

impl<'a, T> AssertLessThanIoColsMut<'a, T> {
    pub fn from_slice(slc: &'a mut [T]) -> Self {
        let (x, rest) = slc.split_first_mut().unwrap();
        let (y, _) = rest.split_first_mut().unwrap();

        Self { x, y }
    }
}

pub struct AssertLessThanAuxColsMut<'a, T> {
    pub lower_decomp: &'a mut [T],
}

impl<'a, T> AssertLessThanAuxColsMut<'a, T> {
    pub fn from_slice(slc: &'a mut [T]) -> Self {
        Self { lower_decomp: slc }
    }
}

pub struct AssertLessThanColsMut<'a, T> {
    pub io: AssertLessThanIoColsMut<'a, T>,
    pub aux: AssertLessThanAuxColsMut<'a, T>,
}

impl<'a, T> AssertLessThanColsMut<'a, T> {
    pub fn from_slice(slc: &'a mut [T]) -> Self {
        let (io, aux) = slc.split_at_mut(2);

        let io = AssertLessThanIoColsMut::from_slice(io);
        let aux = AssertLessThanAuxColsMut::from_slice(aux);

        Self { io, aux }
    }
}
