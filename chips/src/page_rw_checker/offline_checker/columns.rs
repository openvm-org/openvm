use crate::{
    is_equal_vec::columns::IsEqualVecAuxCols, is_less_than_tuple::columns::IsLessThanTupleAuxCols,
};

use super::OfflineChecker;

#[derive(Debug)]
pub struct OfflineCheckerCols<T> {
    /// this bit indicates if this row comes from the initial page
    pub is_initial: T,
    /// this bit indicates if this row should go to the final page (last row for the index)
    pub is_final: T,
    /// this bit indicates if this row refers to an internal operation
    pub is_internal: T,

    /// this is just is_final * 3 (used for interactions)
    pub is_final_x3: T,

    /// timestamp for the operation
    pub clk: T,
    /// the row of the page: starts with is_alloc bit, then index, then data
    pub page_row: Vec<T>,
    /// 0 for read, 1 for write
    pub op_type: T,

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
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        is_initial: T,
        is_final: T,
        is_internal: T,
        is_final_x3: T,
        clk: T,
        page_row: Vec<T>,
        op_type: T,
        same_idx: T,
        same_data: T,
        lt_bit: T,
        is_extra: T,
        is_equal_idx_aux: IsEqualVecAuxCols<T>,
        is_equal_data_aux: IsEqualVecAuxCols<T>,
        lt_aux: IsLessThanTupleAuxCols<T>,
    ) -> Self {
        Self {
            is_initial,
            is_final,
            is_internal,
            is_final_x3,
            clk,
            page_row,
            op_type,
            same_idx,
            same_data,
            lt_bit,
            is_extra,
            is_equal_idx_aux,
            is_equal_data_aux,
            lt_aux,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![
            self.is_initial.clone(),
            self.is_final.clone(),
            self.is_internal.clone(),
            self.is_final_x3.clone(),
            self.clk.clone(),
        ];
        flattened.extend(self.page_row.clone());
        flattened.extend(vec![
            self.op_type.clone(),
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
        let page_width = oc.page_width();

        Self {
            is_initial: slc[0].clone(),
            is_final: slc[1].clone(),
            is_internal: slc[2].clone(),
            is_final_x3: slc[3].clone(),
            clk: slc[4].clone(),
            page_row: slc[5..5 + page_width].to_vec(),
            op_type: slc[5 + page_width].clone(),
            same_idx: slc[6 + page_width].clone(),
            same_data: slc[7 + page_width].clone(),
            lt_bit: slc[8 + page_width].clone(),
            is_extra: slc[9 + page_width].clone(),
            is_equal_idx_aux: IsEqualVecAuxCols::from_slice(
                &slc[10 + page_width..10 + page_width + 2 * oc.idx_len],
                oc.idx_len,
            ),
            is_equal_data_aux: IsEqualVecAuxCols::from_slice(
                &slc[10 + page_width + 2 * oc.idx_len
                    ..10 + page_width + 2 * oc.idx_len + 2 * oc.data_len],
                oc.data_len,
            ),
            lt_aux: IsLessThanTupleAuxCols::from_slice(
                &slc[10 + page_width + 2 * oc.idx_len + 2 * oc.data_len..],
                oc.idx_clk_limb_bits.clone(),
                oc.idx_decomp,
                oc.idx_len + 1,
            ),
        }
    }
}
