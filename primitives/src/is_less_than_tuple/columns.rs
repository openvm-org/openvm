use afs_derive::AlignedBorrow;

use crate::{
    is_equal_vec::columns::{IsEqualVecAuxCols, IsEqualVecAuxColsMut},
    is_less_than::columns::{IsLessThanAuxCols, IsLessThanAuxColsMut},
};

use super::IsLessThanTupleAir;

#[derive(Default, Debug, AlignedBorrow)]
pub struct IsLessThanTupleIoCols<T> {
    pub x: Vec<T>,
    pub y: Vec<T>,
    pub tuple_less_than: T,
}

impl<T: Clone> IsLessThanTupleIoCols<T> {
    pub fn from_slice(slc: &[T], tuple_len: usize) -> Self {
        Self {
            x: slc[0..tuple_len].to_vec(),
            y: slc[tuple_len..2 * tuple_len].to_vec(),
            tuple_less_than: slc[2 * tuple_len].clone(),
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![];
        flattened.extend_from_slice(&self.x);
        flattened.extend_from_slice(&self.y);
        flattened.push(self.tuple_less_than.clone());
        flattened
    }

    pub fn width(tuple_len: usize) -> usize {
        tuple_len + tuple_len + 1
    }
}

#[derive(Debug, Clone)]
pub struct IsLessThanTupleAuxCols<T> {
    pub less_than: Vec<T>,
    pub less_than_aux: Vec<IsLessThanAuxCols<T>>,
    pub is_equal_vec_aux: IsEqualVecAuxCols<T>,
    // TODO: I moved the following column to be its own field instead of being a part of the
    // IsEqualVecAuxCols field (to align with the new IsEqualVec interface so it can be used as a SubAir),
    // but I think it can be removed entirely from the AIR
    pub is_equal_out: T,
    pub less_than_cumulative: Vec<T>,
}

impl<T: Clone> IsLessThanTupleAuxCols<T> {
    pub fn from_slice(slc: &[T], lt_chip: &IsLessThanTupleAir) -> Self {
        let tuple_len = lt_chip.tuple_len();

        let mut iter = slc.iter().cloned();
        let mut take = |n: usize| iter.by_ref().take(n).collect::<Vec<T>>();

        let less_than = take(tuple_len);

        let mut less_than_aux: Vec<IsLessThanAuxCols<T>> = vec![];
        for air in lt_chip.is_less_than_airs.iter() {
            let cur_width = IsLessThanAuxCols::<T>::width(air);
            let less_than_col = IsLessThanAuxCols::from_slice(&take(cur_width));
            less_than_aux.push(less_than_col);
        }

        let prods = take(tuple_len - 1);
        let invs = take(tuple_len);
        let is_equal_out = take(1).remove(0);
        let is_equal_vec_aux = IsEqualVecAuxCols { prods, invs };

        let less_than_cumulative = take(tuple_len);

        Self {
            less_than,
            less_than_aux,
            is_equal_vec_aux,
            is_equal_out,
            less_than_cumulative,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![];

        flattened.extend_from_slice(&self.less_than);

        for i in 0..self.less_than_aux.len() {
            flattened.extend_from_slice(&self.less_than_aux[i].flatten());
        }

        flattened.extend_from_slice(&self.is_equal_vec_aux.prods);
        flattened.extend_from_slice(&self.is_equal_vec_aux.invs);
        flattened.push(self.is_equal_out.clone());

        flattened.extend_from_slice(&self.less_than_cumulative);

        flattened
    }

    pub fn width(lt_air: &IsLessThanTupleAir) -> usize {
        let mut width = 2 * lt_air.tuple_len();
        for air in lt_air.is_less_than_airs.iter() {
            width += IsLessThanAuxCols::<T>::width(air);
        }
        width += IsEqualVecAuxCols::<T>::width(lt_air.tuple_len()) + 1;

        width
    }
}

#[derive(Debug)]
pub struct IsLessThanTupleCols<T> {
    pub io: IsLessThanTupleIoCols<T>,
    pub aux: IsLessThanTupleAuxCols<T>,
}

impl<T: Clone> IsLessThanTupleCols<T> {
    pub fn from_slice(slc: &[T], lt_air: &IsLessThanTupleAir) -> Self {
        let tuple_len = lt_air.tuple_len();

        let io = IsLessThanTupleIoCols::from_slice(&slc[..2 * tuple_len + 1], tuple_len);
        let aux = IsLessThanTupleAuxCols::from_slice(&slc[2 * tuple_len + 1..], lt_air);

        Self { io, aux }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = self.io.flatten();
        flattened.extend(self.aux.flatten());
        flattened
    }

    pub fn width(lt_air: &IsLessThanTupleAir) -> usize {
        IsLessThanTupleIoCols::<T>::width(lt_air.tuple_len())
            + IsLessThanTupleAuxCols::<T>::width(lt_air)
    }
}

pub struct IsLessThanTupleIoColsMut<'a, T> {
    pub x: &'a mut [T],
    pub y: &'a mut [T],
    pub tuple_less_than: &'a mut T,
}

impl<'a, T> IsLessThanTupleIoColsMut<'a, T> {
    pub fn from_slice(slc: &'a mut [T], lt_air: &IsLessThanTupleAir) -> Self {
        let (x, rest) = slc.split_at_mut(lt_air.tuple_len());
        let (tuple_less_than, y) = rest.split_last_mut().unwrap();

        Self {
            x,
            y,
            tuple_less_than,
        }
    }
}

pub struct IsLessThanTupleAuxColsMut<'a, T> {
    pub less_than: &'a mut [T],
    pub less_than_aux: Vec<IsLessThanAuxColsMut<'a, T>>,
    pub is_equal_vec_aux: IsEqualVecAuxColsMut<'a, T>,
    pub is_equal_out: &'a mut T,
    pub less_than_cumulative: &'a mut [T],
}

impl<'a, T> IsLessThanTupleAuxColsMut<'a, T> {
    pub fn from_slice(slc: &'a mut [T], lt_chip: &IsLessThanTupleAir) -> Self {
        let tuple_len = lt_chip.tuple_len();

        let (less_than, mut rest) = slc.split_at_mut(tuple_len);

        let mut less_than_aux: Vec<IsLessThanAuxColsMut<'a, T>> =
            Vec::with_capacity(lt_chip.tuple_len());
        for air in lt_chip.is_less_than_airs.iter() {
            let cur_width = IsLessThanAuxCols::<T>::width(air);
            let (cur_slc, new_rest) = rest.split_at_mut(cur_width);
            let less_than_col = IsLessThanAuxColsMut::from_slice(cur_slc);
            less_than_aux.push(less_than_col);

            rest = new_rest;
        }

        let (prods, rest) = rest.split_at_mut(tuple_len - 1);
        let (invs, rest) = rest.split_at_mut(tuple_len);

        let (is_equal_out, less_than_cumulative) = rest.split_first_mut().unwrap();

        let is_equal_vec_aux = IsEqualVecAuxColsMut { prods, invs };

        Self {
            less_than,
            less_than_aux,
            is_equal_vec_aux,
            is_equal_out,
            less_than_cumulative,
        }
    }
}

pub struct IsLessThanTupleColsMut<'a, T> {
    pub io: IsLessThanTupleIoColsMut<'a, T>,
    pub aux: IsLessThanTupleAuxColsMut<'a, T>,
}

impl<'a, T> IsLessThanTupleColsMut<'a, T> {
    pub fn from_slice(slc: &'a mut [T], lt_air: &IsLessThanTupleAir) -> Self {
        let (io, aux) = slc.split_at_mut(2 * lt_air.tuple_len() + 1);

        let io = IsLessThanTupleIoColsMut::from_slice(io, lt_air);
        let aux = IsLessThanTupleAuxColsMut::from_slice(aux, lt_air);

        Self { io, aux }
    }
}
