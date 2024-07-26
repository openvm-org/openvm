use derive_new::new;

use crate::{
    is_equal_vec::columns::IsEqualVecAuxCols, is_less_than_tuple::columns::IsLessThanTupleAuxCols,
};

use super::OfflineChecker;

#[allow(clippy::too_many_arguments)]
#[derive(Debug, Clone, new)]
pub struct OfflineCheckerCols<T> {
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

    /// this bit indicates if (idx, clk) is strictly more than the one in the previous row
    pub lt_bit: T,
    /// a bit to indicate if this is a valid operation row
    pub is_valid: T,
    /// a bit to indicate whether this operation row should be received
    pub is_receive: T,

    /// auxiliary columns used for same_idx
    pub is_equal_idx_aux: IsEqualVecAuxCols<T>,
    /// auxiliary columns to check proper sorting
    pub lt_aux: IsLessThanTupleAuxCols<T>,
}

impl<T> OfflineCheckerCols<T>
where
    T: Clone,
{
    pub fn flatten(&self, buf: &mut [T], start: usize) -> usize {
        buf[start] = self.clk.clone();
        buf[start + 1..start + 1 + self.idx.len()].clone_from_slice(&self.idx);
        buf[start + 1 + self.idx.len()..start + 1 + self.idx.len() + self.data.len()]
            .clone_from_slice(&self.data);
        buf[start + 1 + self.idx.len() + self.data.len()] = self.op_type.clone();
        buf[start + 2 + self.idx.len() + self.data.len()] = self.same_idx.clone();
        buf[start + 3 + self.idx.len() + self.data.len()] = self.lt_bit.clone();
        buf[start + 4 + self.idx.len() + self.data.len()] = self.is_valid.clone();
        buf[start + 5 + self.idx.len() + self.data.len()] = self.is_receive.clone();
        let mut cum_len = 6 + self.idx.len() + self.data.len();
        let is_eq_aux_len = self.is_equal_idx_aux.flatten(buf, start + cum_len);
        cum_len += is_eq_aux_len;
        let lt_aux_len = self.lt_aux.flatten(buf, start + cum_len);
        cum_len += lt_aux_len;
        cum_len
    }

    pub fn from_slice(slc: &[T], oc: &OfflineChecker) -> Self {
        assert!(slc.len() == oc.air_width());
        let idx_len = oc.idx_len;
        let data_len = oc.data_len;

        Self {
            clk: slc[0].clone(),
            idx: slc[1..1 + idx_len].to_vec(),
            data: slc[1 + idx_len..1 + idx_len + data_len].to_vec(),
            op_type: slc[1 + idx_len + data_len].clone(),
            same_idx: slc[2 + idx_len + data_len].clone(),
            lt_bit: slc[3 + idx_len + data_len].clone(),
            is_valid: slc[4 + idx_len + data_len].clone(),
            is_receive: slc[5 + idx_len + data_len].clone(),
            is_equal_idx_aux: IsEqualVecAuxCols::from_slice(
                &slc[6 + idx_len + data_len..5 + 3 * idx_len + data_len],
                idx_len,
            ),
            lt_aux: IsLessThanTupleAuxCols::from_slice(
                &slc[5 + 3 * idx_len + data_len..],
                &oc.lt_tuple_air,
            ),
        }
    }
}
