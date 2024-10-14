use afs_derive::AlignedBorrow;

use crate::{
    is_less_than_tuple::{columns::IsLessThanTupleAuxCols, IsLessThanTupleAir},
    var_range::bus::VariableRangeCheckerBus,
};

// Since AssertSortedChip contains a LessThanChip subchip, a subset of the columns are those of the
// LessThanChip
#[derive(AlignedBorrow)]
pub struct AssertSortedCols<T> {
    pub key: Vec<T>,
    pub less_than_next_key: T,
    pub is_less_than_tuple_aux: IsLessThanTupleAuxCols<T>,
}

impl<T: Clone> AssertSortedCols<T> {
    pub fn from_slice(slc: &[T], limb_bits: &[usize], max_range_bits: usize) -> Self {
        let key_vec_len = limb_bits.len();

        let mut curr_start_idx = 0;
        let mut curr_end_idx = key_vec_len;

        // the first key_vec_len elements are the key itself
        let key = slc[curr_start_idx..curr_end_idx].to_vec();

        curr_start_idx = curr_end_idx;
        curr_end_idx += 1;

        // the next element is the indicator for whether the key is less than the next key
        let less_than_next_key = slc[curr_start_idx].clone();
        curr_start_idx = curr_end_idx;

        let lt_chip = IsLessThanTupleAir::new(
            VariableRangeCheckerBus::new(0, max_range_bits),
            limb_bits.to_vec(),
        );
        let is_less_than_tuple_aux =
            IsLessThanTupleAuxCols::from_slice(&slc[curr_start_idx..], &lt_chip);

        Self {
            key,
            less_than_next_key,
            is_less_than_tuple_aux,
        }
    }

    pub fn get_width(limb_bits: &[usize], max_range_bits: usize) -> usize {
        limb_bits.len()
            + 1
            + IsLessThanTupleAuxCols::<T>::width(&IsLessThanTupleAir::new(
                VariableRangeCheckerBus::new(0, max_range_bits),
                limb_bits.to_vec(),
            ))
    }
}
