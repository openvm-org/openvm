use afs_derive::AlignedBorrow;
use p3_air::BaseAir;

use crate::is_less_than::{columns::IsLessThanAuxCols, IsLessThanAir};

use super::SumAir;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct SumCols<T> {
    pub key: T,
    pub value: T,
    pub partial_sum: T,
    pub is_final: T,
    pub is_lt_aux_cols: IsLessThanAuxCols<T>,
}

impl<T: Clone> SumCols<T> {
    pub fn from_slice(slc: &[T], lt_air: &IsLessThanAir) -> Self {
        let cols = SumCols::<usize>::index_map(lt_air);
        let key = slc[cols.key].clone();
        let value = slc[cols.value].clone();
        let partial_sum = slc[cols.partial_sum].clone();
        let is_final = slc[cols.is_final].clone();

        let is_lt_aux_cols = IsLessThanAuxCols::<T>::from_slice(&slc[cols.is_lt_aux_cols.lower..]);
        SumCols {
            key,
            value,
            partial_sum,
            is_final,
            is_lt_aux_cols,
        }
    }

    pub fn index_map(lt_air: &IsLessThanAir) -> SumCols<usize> {
        let num_aux_cols = IsLessThanAuxCols::<usize>::width(lt_air);
        SumCols {
            key: 0,
            value: 1,
            partial_sum: 2,
            is_final: 3,
            is_lt_aux_cols: IsLessThanAuxCols {
                lower: 4,
                lower_decomp: (5..5 + num_aux_cols - 1).collect(),
            },
        }
    }
}

impl<T: Clone> BaseAir<T> for SumAir {
    fn width(&self) -> usize {
        4 + IsLessThanAuxCols::<T>::width(&self.is_lt_air)
    }
}
