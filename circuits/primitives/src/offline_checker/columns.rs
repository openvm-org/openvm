use derive_new::new;

use super::OfflineChecker;
use crate::{
    is_equal_vec::columns::{IsEqualVecAuxCols, IsEqualVecAuxColsMut},
    is_less_than_tuple::columns::{IsLessThanTupleAuxCols, IsLessThanTupleAuxColsMut},
};

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

impl<T: Clone> OfflineCheckerCols<T> {
    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![self.clk.clone()];
        flattened.extend(self.idx.clone());
        flattened.extend(self.data.clone());
        flattened.extend(vec![
            self.op_type.clone(),
            self.same_idx.clone(),
            self.lt_bit.clone(),
            self.is_valid.clone(),
            self.is_receive.clone(),
        ]);

        flattened.extend(self.is_equal_idx_aux.flatten());
        flattened.extend(self.lt_aux.flatten());

        flattened
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

impl<T> OfflineCheckerCols<T> {
    pub fn width(oc: &OfflineChecker) -> usize {
        6 + oc.idx_len
            + oc.data_len
            + IsEqualVecAuxCols::<usize>::width(oc.idx_len)
            + IsLessThanTupleAuxCols::<usize>::width(&oc.lt_tuple_air)
    }
}

pub struct OfflineCheckerColsMut<'a, T> {
    /// timestamp for the operation
    pub clk: &'a mut T,
    /// idx
    pub idx: &'a mut [T],
    /// data
    pub data: &'a mut [T],
    /// default: 0 (read) and 1 (write), can have more e.g. delete
    pub op_type: &'a mut T,

    /// this bit indicates if the idx matches the one in the previous row
    /// (should be 0 in first row)
    pub same_idx: &'a mut T,

    /// this bit indicates if (idx, clk) is strictly more than the one in the previous row
    pub lt_bit: &'a mut T,
    /// a bit to indicate if this is a valid operation row
    pub is_valid: &'a mut T,
    /// a bit to indicate whether this operation row should be received
    pub is_receive: &'a mut T,

    /// auxiliary columns used for same_idx
    pub is_equal_idx_aux: IsEqualVecAuxColsMut<'a, T>,
    /// auxiliary columns to check proper sorting
    pub lt_aux: IsLessThanTupleAuxColsMut<'a, T>,
}

impl<'a, T> OfflineCheckerColsMut<'a, T> {
    pub fn from_slice(slc: &'a mut [T], oc: &OfflineChecker) -> Self {
        assert!(slc.len() == oc.air_width());
        let idx_len = oc.idx_len;
        let data_len = oc.data_len;

        let (clk, rest) = slc.split_first_mut().unwrap();
        let (idx, rest) = rest.split_at_mut(idx_len);
        let (data, rest) = rest.split_at_mut(data_len);
        let (op_type, rest) = rest.split_first_mut().unwrap();
        let (same_idx, rest) = rest.split_first_mut().unwrap();
        let (lt_bit, rest) = rest.split_first_mut().unwrap();
        let (is_valid, rest) = rest.split_first_mut().unwrap();
        let (is_receive, rest) = rest.split_first_mut().unwrap();

        let is_equal_aux_width = IsEqualVecAuxCols::<usize>::width(idx_len);
        let (e_aux, lt_aux) = rest.split_at_mut(is_equal_aux_width);

        let (is_equal_idx_aux, lt_aux) = (
            IsEqualVecAuxColsMut::from_slice(e_aux, &oc.is_equal_idx_air),
            IsLessThanTupleAuxColsMut::from_slice(lt_aux, &oc.lt_tuple_air),
        );

        Self {
            clk,
            idx,
            data,
            op_type,
            same_idx,
            lt_bit,
            is_valid,
            is_receive,
            is_equal_idx_aux,
            lt_aux,
        }
    }
}
