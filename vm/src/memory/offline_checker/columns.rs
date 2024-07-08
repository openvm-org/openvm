use afs_chips::{
    is_equal::columns::IsEqualAuxCols, is_equal_vec::columns::IsEqualVecAuxCols,
    is_less_than_tuple::columns::IsLessThanTupleAuxCols,
};

use super::OfflineChecker;

#[derive(Debug)]
pub struct OfflineCheckerCols<T> {
    /// timestamp for the operation
    pub clk: T,
    /// address space, pointer, data
    pub mem_row: Vec<T>,
    /// 0 for read, 1 for write
    pub op_type: T,

    /// this bit indicates if the address space matches the one in the previous row (should be 0 in first row)
    pub same_addr_space: T,
    /// this bit indicates if the pointer matches the one in the previous row (should be 0 in first row)
    pub same_pointer: T,
    /// this bit indicates if the address matches the one in the previous row, i.e. same_addr_space and same_pointer
    /// (should be 0 in first row)
    pub same_addr: T,
    /// this bit indicates if the data matches the one in the previous row (should be 0 in first row)
    pub same_data: T,

    /// this bit indicates if (addr_space, pointer, clk) is strictly more than the one in the previous row
    pub lt_bit: T,
    /// a bit to indicate if this is a valid operation row
    pub is_valid: T,

    /// auxiliary columns used for same_addr_space
    pub is_equal_addr_space_aux: IsEqualAuxCols<T>,
    /// auxiliary columns used for same_pointer
    pub is_equal_pointer_aux: IsEqualAuxCols<T>,
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
        clk: T,
        mem_row: Vec<T>,
        op_type: T,
        same_addr_space: T,
        same_pointer: T,
        same_addr: T,
        same_data: T,
        lt_bit: T,
        is_valid: T,
        is_equal_addr_space_aux: IsEqualAuxCols<T>,
        is_equal_pointer_aux: IsEqualAuxCols<T>,
        is_equal_data_aux: IsEqualVecAuxCols<T>,
        lt_aux: IsLessThanTupleAuxCols<T>,
    ) -> Self {
        Self {
            clk,
            mem_row,
            op_type,
            same_addr_space,
            same_pointer,
            same_addr,
            same_data,
            lt_bit,
            is_valid,
            is_equal_addr_space_aux,
            is_equal_pointer_aux,
            is_equal_data_aux,
            lt_aux,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![self.clk.clone()];
        flattened.extend(self.mem_row.clone());
        flattened.extend(vec![
            self.op_type.clone(),
            self.same_addr_space.clone(),
            self.same_pointer.clone(),
            self.same_addr.clone(),
            self.same_data.clone(),
            self.lt_bit.clone(),
            self.is_valid.clone(),
        ]);

        flattened.extend(self.is_equal_addr_space_aux.flatten());
        flattened.extend(self.is_equal_pointer_aux.flatten());
        flattened.extend(self.is_equal_data_aux.flatten());
        flattened.extend(self.lt_aux.flatten());

        flattened
    }

    pub fn from_slice<const WORD_SIZE: usize>(slc: &[T], oc: &OfflineChecker<WORD_SIZE>) -> Self {
        assert!(slc.len() == oc.air_width());
        let mem_width = oc.mem_width();

        Self {
            clk: slc[0].clone(),
            mem_row: slc[1..1 + mem_width].to_vec(),
            op_type: slc[1 + mem_width].clone(),
            same_addr_space: slc[2 + mem_width].clone(),
            same_pointer: slc[3 + mem_width].clone(),
            same_addr: slc[4 + mem_width].clone(),
            same_data: slc[5 + mem_width].clone(),
            lt_bit: slc[6 + mem_width].clone(),
            is_valid: slc[7 + mem_width].clone(),
            is_equal_addr_space_aux: IsEqualAuxCols::from_slice(&slc[8 + mem_width..9 + mem_width]),
            is_equal_pointer_aux: IsEqualAuxCols::from_slice(&slc[9 + mem_width..10 + mem_width]),
            is_equal_data_aux: IsEqualVecAuxCols::from_slice(
                &slc[10 + mem_width..10 + mem_width + 2 * WORD_SIZE],
                WORD_SIZE,
            ),
            lt_aux: IsLessThanTupleAuxCols::from_slice(
                &slc[10 + mem_width + 2 * WORD_SIZE..],
                oc.addr_clk_limb_bits.clone(),
                oc.decomp,
                3,
            ),
        }
    }
}
