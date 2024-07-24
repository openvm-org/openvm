use afs_derive::AlignedBorrow;

use crate::{is_equal_vec::columns::IsEqualVecAuxCols, is_less_than::columns::IsLessThanAuxCols};

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

    pub fn get_width(tuple_len: usize) -> usize {
        tuple_len + tuple_len + 1
    }
}

#[derive(Debug, Clone)]
pub struct IsLessThanTupleAuxCols<T> {
    pub less_than: Vec<T>,
    pub less_than_aux: Vec<IsLessThanAuxCols<T>>,
    pub is_equal_vec_aux: IsEqualVecAuxCols<T>,
    pub less_than_cumulative: Vec<T>,
}

impl<T: Clone> IsLessThanTupleAuxCols<T> {
    pub fn from_slice(slc: &[T], limb_bits: Vec<usize>, decomp: usize, tuple_len: usize) -> Self {
        assert!(limb_bits.len() == tuple_len);

        let mut curr_start_idx = 0;
        let mut curr_end_idx = tuple_len;

        let less_than = slc[curr_start_idx..curr_end_idx].to_vec();

        curr_start_idx = curr_end_idx;
        curr_end_idx += tuple_len;

        // get the lower bits for each 2^limb_bits[i] + y[i] - x[i] - 1
        let lower_vec = slc[curr_start_idx..curr_end_idx].to_vec();

        // get the lower bits decompositions
        let mut lower_decomp_vec: Vec<Vec<T>> = vec![];

        for &limb_bit in limb_bits.iter() {
            let num_limbs = (limb_bit + decomp - 1) / decomp;
            curr_start_idx = curr_end_idx;
            curr_end_idx += num_limbs + 1;

            let mut lower_bits_curr: Vec<T> = vec![];

            for j in 0..(num_limbs + 1) {
                lower_bits_curr.push(slc[curr_start_idx + j].clone());
            }

            lower_decomp_vec.push(lower_bits_curr);
        }

        curr_start_idx = curr_end_idx;
        curr_end_idx += tuple_len;

        // generate the less_than_aux columns
        let mut less_than_aux: Vec<IsLessThanAuxCols<T>> = vec![];
        for i in 0..tuple_len {
            let less_than_col = IsLessThanAuxCols {
                lower: lower_vec[i].clone(),
                lower_decomp: lower_decomp_vec[i].clone(),
            };

            less_than_aux.push(less_than_col);
        }

        // prods[i] indicates whether x[i] == y[i] up to the i-th index
        let prods = slc[curr_start_idx..curr_end_idx].to_vec();

        curr_start_idx = curr_end_idx;
        curr_end_idx += tuple_len;

        // get invs
        let invs = slc[curr_start_idx..curr_end_idx].to_vec();

        curr_start_idx = curr_end_idx;
        curr_end_idx += tuple_len;

        let is_equal_vec_aux = IsEqualVecAuxCols { prods, invs };

        let less_than_cumulative = slc[curr_start_idx..curr_end_idx].to_vec();

        Self {
            less_than,
            less_than_aux,
            is_equal_vec_aux,
            less_than_cumulative,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![];

        flattened.extend_from_slice(&self.less_than);

        for i in 0..self.less_than_aux.len() {
            flattened.push(self.less_than_aux[i].lower.clone());
        }

        for i in 0..self.less_than_aux.len() {
            flattened.extend_from_slice(&self.less_than_aux[i].lower_decomp);
        }

        flattened.extend_from_slice(&self.is_equal_vec_aux.prods);
        flattened.extend_from_slice(&self.is_equal_vec_aux.invs);

        flattened.extend_from_slice(&self.less_than_cumulative);

        flattened
    }
}

impl<T> IsLessThanTupleAuxCols<T> {
    pub fn get_width(limb_bits: Vec<usize>, decomp: usize, tuple_len: usize) -> usize {
        let mut width = 0;
        // for the less than indicator
        width += tuple_len;
        // for the lowers
        width += tuple_len;
        // for the decomposed lowers
        for &limb_bit in limb_bits.iter() {
            let num_limbs = (limb_bit + decomp - 1) / decomp;
            width += num_limbs + 1;
        }
        // for the prods
        width += tuple_len;
        // for the invs
        width += tuple_len;
        // for the cumulative less_than
        width += tuple_len;

        width
    }
}

#[derive(Debug)]
pub struct IsLessThanTupleCols<T> {
    pub io: IsLessThanTupleIoCols<T>,
    pub aux: IsLessThanTupleAuxCols<T>,
}

impl<T: Clone> IsLessThanTupleCols<T> {
    pub fn from_slice(slc: &[T], limb_bits: Vec<usize>, decomp: usize, tuple_len: usize) -> Self {
        let io = IsLessThanTupleIoCols::from_slice(&slc[..2 * tuple_len + 1], tuple_len);
        let aux = IsLessThanTupleAuxCols::from_slice(
            &slc[2 * tuple_len + 1..],
            limb_bits,
            decomp,
            tuple_len,
        );

        Self { io, aux }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = self.io.flatten();
        flattened.extend(self.aux.flatten());
        flattened
    }

    pub fn get_width(limb_bits: Vec<usize>, decomp: usize, tuple_len: usize) -> usize {
        IsLessThanTupleIoCols::<T>::get_width(tuple_len)
            + IsLessThanTupleAuxCols::<T>::get_width(limb_bits, decomp, tuple_len)
    }
}
