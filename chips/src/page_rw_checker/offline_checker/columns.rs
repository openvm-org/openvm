use crate::{
    is_equal_vec::columns::IsEqualVecAuxCols, is_less_than_tuple::columns::IsLessThanTupleAuxCols,
};

use super::OfflineChecker;

#[allow(clippy::too_many_arguments)]
#[derive(Debug, derive_new::new)]
pub struct OfflineCheckerCols<T> {
    /// this bit indicates if this row comes from the initial page
    pub is_initial: T,
    /// this bit indicates if this is the final row of an idx and that it should be sent to the final chip
    pub is_final_write: T,
    /// this bit indicates if this is the final row of an idx and that it that it was deleted (shouldn't be sent to the final chip)
    pub is_final_delete: T,
    /// this bit indicates if this row refers to an internal operation
    pub is_internal: T,

    /// this is just is_final_write * 3 (used for interactions)
    pub is_final_write_x3: T,

    /// timestamp for the operation
    pub clk: T,
    /// the row of the page without the is_alloc bit: idx and data only
    pub page_row: Vec<T>,
    /// 0 for read, 1 for write, 2 for delete
    pub op_type: T,
    /// 1 if the operation is a read, 0 otherwise
    pub is_read: T,
    /// 1 if the operation is a write, 0 otherwise
    pub is_write: T,
    /// 1 if the operation is a delete, 0 otherwise
    pub is_delete: T,

    /// this bit indicates if the index matches the one in the previous row (should be 0 in first row)
    pub same_idx: T,
    /// this bit indicates if the data matches the one in the previous row (should be 0 in first row)
    pub same_data: T,
    /// this bit indicates if (idx, clk) is strictly more than the one in the previous row
    pub lt_bit: T,
    /// a bit to indicate if this is an extra row that should be ignored
    pub is_extra: T,

    /// auxiliary columns used for same_idx
    pub is_equal_idx_aux: IsEqualVecAuxCols<T>,
    /// auxiliary columns used for same_data
    pub is_equal_data_aux: IsEqualVecAuxCols<T>,
    /// auxiliary columns to check proper sorting
    pub lt_aux: IsLessThanTupleAuxCols<T>,
}

impl<T> OfflineCheckerCols<T>
where
    T: Clone,
{
    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![
            self.is_initial.clone(),
            self.is_final_write.clone(),
            self.is_final_delete.clone(),
            self.is_internal.clone(),
            self.is_final_write_x3.clone(),
            self.clk.clone(),
        ];
        flattened.extend(self.page_row.clone());
        flattened.extend(vec![
            self.op_type.clone(),
            self.is_read.clone(),
            self.is_write.clone(),
            self.is_delete.clone(),
            self.same_idx.clone(),
            self.same_data.clone(),
            self.lt_bit.clone(),
            self.is_extra.clone(),
        ]);

        flattened.extend(self.is_equal_idx_aux.flatten());
        flattened.extend(self.is_equal_data_aux.flatten());
        flattened.extend(self.lt_aux.flatten());

        flattened
    }

    pub fn from_slice(slc: &[T], oc: &OfflineChecker) -> Self {
        assert!(slc.len() == oc.air_width());
        let page_row_width = oc.idx_len + oc.data_len;

        Self {
            is_initial: slc[0].clone(),
            is_final_write: slc[1].clone(),
            is_final_delete: slc[2].clone(),
            is_internal: slc[3].clone(),
            is_final_write_x3: slc[4].clone(),
            clk: slc[5].clone(),
            page_row: slc[6..6 + page_row_width].to_vec(),
            op_type: slc[6 + page_row_width].clone(),
            is_read: slc[7 + page_row_width].clone(),
            is_write: slc[8 + page_row_width].clone(),
            is_delete: slc[9 + page_row_width].clone(),
            same_idx: slc[10 + page_row_width].clone(),
            same_data: slc[11 + page_row_width].clone(),
            lt_bit: slc[12 + page_row_width].clone(),
            is_extra: slc[13 + page_row_width].clone(),
            is_equal_idx_aux: IsEqualVecAuxCols::from_slice(
                &slc[14 + page_row_width..14 + page_row_width + 2 * oc.idx_len],
                oc.idx_len,
            ),
            is_equal_data_aux: IsEqualVecAuxCols::from_slice(
                &slc[14 + page_row_width + 2 * oc.idx_len
                    ..14 + page_row_width + 2 * oc.idx_len + 2 * oc.data_len],
                oc.data_len,
            ),
            lt_aux: IsLessThanTupleAuxCols::from_slice(
                &slc[14 + page_row_width + 2 * oc.idx_len + 2 * oc.data_len..],
                oc.idx_clk_limb_bits.clone(),
                oc.idx_decomp,
                oc.idx_len + 1,
            ),
        }
    }

    pub fn width(oc: &OfflineChecker) -> usize {
        14 + oc.idx_len
            + oc.data_len
            + 2 * (oc.idx_len + oc.data_len)
            + IsLessThanTupleAuxCols::<usize>::get_width(
                oc.idx_clk_limb_bits.clone(),
                oc.idx_decomp,
                oc.idx_len + 1,
            )
    }
}
