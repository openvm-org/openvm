use afs_derive::AlignedBorrow;

use crate::{is_equal::columns::IsEqualAuxCols, is_less_than_bits::columns::IsLessThanBitsAuxCols};

#[derive(Default, AlignedBorrow)]
pub struct IsLessThanTupleBitsIOCols<T> {
    pub x: Vec<T>,
    pub y: Vec<T>,
    pub tuple_less_than: T,
}

impl<T: Clone> IsLessThanTupleBitsIOCols<T> {
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

    pub fn get_width(tuple_len: usize) -> usize {
        tuple_len + tuple_len + 1
    }
}

pub struct IsLessThanTupleBitsAuxCols<T> {
    pub less_than: Vec<T>,
    pub less_than_aux: Vec<IsLessThanBitsAuxCols<T>>,
    pub is_equal: Vec<T>,
    pub is_equal_aux: Vec<IsEqualAuxCols<T>>,

    pub less_than_cumulative: Vec<T>,
}

impl<T: Clone> IsLessThanTupleBitsAuxCols<T> {
    pub fn from_slice(slc: &[T], limb_bits: Vec<usize>, tuple_len: usize) -> Self {
        let mut curr_start_idx = 0;
        let mut curr_end_idx = tuple_len;

        let less_than = slc[curr_start_idx..curr_end_idx].to_vec();

        let mut less_than_aux = vec![];
        for &limb_bit in limb_bits.iter() {
            curr_start_idx = curr_end_idx;
            curr_end_idx += IsLessThanBitsAuxCols::<T>::get_width(limb_bit);
            let less_than_aux_col =
                IsLessThanBitsAuxCols::from_slice(&slc[curr_start_idx..curr_end_idx]);
            less_than_aux.push(less_than_aux_col);
        }

        curr_start_idx = curr_end_idx;
        curr_end_idx += tuple_len;

        // get whether y[i] - x[i] == 0
        let is_equal = slc[curr_start_idx..curr_end_idx].to_vec();

        let mut is_equal_aux = vec![];
        for _i in 0..tuple_len {
            curr_start_idx = curr_end_idx;
            curr_end_idx += 1;
            let is_equal_aux_col = IsEqualAuxCols {
                inv: slc[curr_start_idx].clone(),
            };
            is_equal_aux.push(is_equal_aux_col);
        }

        curr_start_idx = curr_end_idx;
        curr_end_idx += tuple_len;

        let less_than_cumulative = slc[curr_start_idx..curr_end_idx].to_vec();

        Self {
            less_than,
            less_than_aux,
            is_equal,
            is_equal_aux,
            less_than_cumulative,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![];

        flattened.extend_from_slice(&self.less_than);

        for i in 0..self.less_than_aux.len() {
            flattened.extend(self.less_than_aux[i].flatten());
        }

        flattened.extend_from_slice(&self.is_equal);

        for i in 0..self.is_equal_aux.len() {
            flattened.push(self.is_equal_aux[i].inv.clone());
        }

        flattened.extend_from_slice(&self.less_than_cumulative);

        flattened
    }

    pub fn get_width(limb_bits: Vec<usize>) -> usize {
        let mut width = 0;

        for &limb_bit in limb_bits.iter() {
            // less than aux cols
            width += IsLessThanBitsAuxCols::<T>::get_width(limb_bit);
            // equal aux col, equal bit, less than bit, cumulative less than bit
            width += 4;
        }

        width
    }
}

pub struct IsLessThanTupleBitsCols<T> {
    pub io: IsLessThanTupleBitsIOCols<T>,
    pub aux: IsLessThanTupleBitsAuxCols<T>,
}

impl<T: Clone> IsLessThanTupleBitsCols<T> {
    pub fn from_slice(slc: &[T], limb_bits: Vec<usize>, tuple_len: usize) -> Self {
        let io = IsLessThanTupleBitsIOCols::from_slice(&slc[..2 * tuple_len + 1], tuple_len);
        let aux =
            IsLessThanTupleBitsAuxCols::from_slice(&slc[2 * tuple_len + 1..], limb_bits, tuple_len);

        Self { io, aux }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = self.io.flatten();
        flattened.extend(self.aux.flatten());
        flattened
    }

    pub fn get_width(limb_bits: Vec<usize>, tuple_len: usize) -> usize {
        IsLessThanTupleBitsIOCols::<T>::get_width(tuple_len)
            + IsLessThanTupleBitsAuxCols::<T>::get_width(limb_bits)
    }
}
