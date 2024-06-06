use afs_derive::AlignedBorrow;

use crate::is_less_than_tuple::columns::IsLessThanTupleAuxCols;

// Since AssertSortedChip contains a LessThanChip subchip, a subset of the columns are those of the
// LessThanChip
#[derive(AlignedBorrow)]
pub struct AssertSortedCols<T> {
    pub key: Vec<T>,
    pub less_than_next_key: T,
    pub is_less_than_tuple_aux: IsLessThanTupleAuxCols<T>,
}

impl<T: Clone> AssertSortedCols<T> {
    pub fn from_slice(slc: &[T], limb_bits: Vec<usize>, decomp: usize, key_vec_len: usize) -> Self {
        let mut curr_start_idx = 0;
        let mut curr_end_idx = key_vec_len;

        // the first key_vec_len elements are the key itself
        let key = slc[curr_start_idx..curr_end_idx].to_vec();

        curr_start_idx = curr_end_idx;
        curr_end_idx += 1;

        // the next element is the indicator for whether the key is less than the next key
        let less_than_next_key = slc[curr_start_idx].clone();
        curr_start_idx = curr_end_idx;

        let is_less_than_tuple_aux = IsLessThanTupleAuxCols::from_slice(
            &slc[curr_start_idx..],
            limb_bits,
            decomp,
            key_vec_len,
        );

        Self {
            key,
            less_than_next_key,
            is_less_than_tuple_aux,
        }
    }

    pub fn get_width(limb_bits: Vec<usize>, decomp: usize, key_vec_len: usize) -> usize {
        let mut width = 0;
        // for the key itself
        width += key_vec_len;

        // for the less than next key indicator
        width += 1;

        // for the less_than indicators
        width += key_vec_len;

        // for the lowers
        width += key_vec_len;

        // for the decomposed lowers
        for &limb_bit in limb_bits.iter() {
            let num_limbs = (limb_bit + decomp - 1) / decomp;
            width += num_limbs + 1;
        }

        // for the is_equal indicators
        width += key_vec_len;

        // for the inverses
        width += key_vec_len;

        // for the cumulative is_equal and less_than
        width += 2 * key_vec_len;

        width
    }
}
