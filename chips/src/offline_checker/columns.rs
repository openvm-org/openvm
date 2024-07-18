use crate::{
    is_equal_vec::columns::IsEqualVecAuxCols, is_less_than_tuple::columns::IsLessThanTupleAuxCols,
};

use super::GeneralOfflineChecker;

#[derive(Debug, Clone)]
pub struct GeneralOfflineCheckerCols<T> {
    /// timestamp for the operation
    pub clk: T,
    /// idx
    pub idx: Vec<T>,
    /// data
    pub data: Vec<T>,
    /// default: 0 (read) and 1 (write), can have more e.g. delete
    pub op_type: T,

    /// this bit indicates if the idx matches the one in the previous row
    /// (should be 0 in first row)
    pub same_idx: T,
    /// this bit indicates if the data matches the one in the previous row (should be 0 in first row)
    pub same_data: T,
    /// this bit indicates if the idx and data match the one in the previous row (should be 0 in first row)
    /// this is used to reduce the degree of a constraint
    pub same_idx_and_data: T,

    /// this bit indicates if (idx, clk) is strictly more than the one in the previous row
    pub lt_bit: T,
    /// a bit to indicate if this is a valid operation row
    pub is_valid: T,

    /// auxiliary columns used for same_addr_space
    pub is_equal_idx_aux: IsEqualVecAuxCols<T>,
    /// auxiliary columns used for same_data
    pub is_equal_data_aux: IsEqualVecAuxCols<T>,
    /// auxiliary columns to check proper sorting
    pub lt_aux: IsLessThanTupleAuxCols<T>,
}

impl<T> GeneralOfflineCheckerCols<T>
where
    T: Clone,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        clk: T,
        idx: Vec<T>,
        data: Vec<T>,
        op_type: T,
        same_idx: T,
        same_data: T,
        same_idx_and_data: T,
        lt_bit: T,
        is_valid: T,
        is_equal_idx_aux: IsEqualVecAuxCols<T>,
        is_equal_data_aux: IsEqualVecAuxCols<T>,
        lt_aux: IsLessThanTupleAuxCols<T>,
    ) -> Self {
        Self {
            clk,
            idx,
            data,
            op_type,
            same_idx,
            same_data,
            same_idx_and_data,
            lt_bit,
            is_valid,
            is_equal_idx_aux,
            is_equal_data_aux,
            lt_aux,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![self.clk.clone()];
        flattened.extend(self.idx.clone());
        flattened.extend(self.data.clone());
        flattened.extend(vec![
            self.op_type.clone(),
            self.same_idx.clone(),
            self.same_data.clone(),
            self.same_idx_and_data.clone(),
            self.lt_bit.clone(),
            self.is_valid.clone(),
        ]);

        flattened.extend(self.is_equal_idx_aux.flatten());
        flattened.extend(self.is_equal_data_aux.flatten());
        flattened.extend(self.lt_aux.flatten());

        flattened
    }

    pub fn from_slice(slc: &[T], oc: &GeneralOfflineChecker) -> Self {
        assert!(slc.len() == oc.air_width());
        let idx_len = oc.idx_len;
        let data_len = oc.data_len;

        Self {
            clk: slc[0].clone(),
            idx: slc[1..1 + idx_len].to_vec(),
            data: slc[1 + idx_len..1 + idx_len + data_len].to_vec(),
            op_type: slc[1 + idx_len + data_len].clone(),
            same_idx: slc[2 + idx_len + data_len].clone(),
            same_data: slc[3 + idx_len + data_len].clone(),
            same_idx_and_data: slc[4 + idx_len + data_len].clone(),
            lt_bit: slc[5 + idx_len + data_len].clone(),
            is_valid: slc[6 + idx_len + data_len].clone(),
            is_equal_idx_aux: IsEqualVecAuxCols::from_slice(
                &slc[7 + idx_len + data_len..6 + 3 * idx_len + data_len],
                idx_len,
            ),
            is_equal_data_aux: IsEqualVecAuxCols::from_slice(
                &slc[6 + 3 * idx_len + data_len..5 + 3 * idx_len + 3 * data_len],
                data_len,
            ),
            lt_aux: IsLessThanTupleAuxCols::from_slice(
                &slc[5 + 3 * idx_len + 3 * data_len..],
                oc.idx_clk_limb_bits.clone(),
                oc.decomp,
                // extra 1 for clk
                idx_len + 1,
            ),
        }
    }
}
