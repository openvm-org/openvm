use afs_derive::AlignedBorrow;

use super::GroupByAir;
use crate::is_equal_vec::columns::IsEqualVecAuxCols;
use crate::is_less_than_tuple::columns::IsLessThanTupleAuxCols;

// Since GroupByChip contains a LessThanChip subchip and an IsEqualVecChip subchip, a subset of the
// columns are those of the LessThanChip and IsEqualVecChip
#[derive(AlignedBorrow)]
pub struct GroupByCols<T> {
    pub page: Vec<T>,
    pub sorted_group_by: Vec<T>,
    pub aggregated: T,
    pub partial_aggregated: T,
    pub is_final: T,
    pub eq_next: T,
    pub is_less_than_tuple_aux: IsLessThanTupleAuxCols<T>,
    pub is_equal_vec_aux: IsEqualVecAuxCols<T>,
}

pub struct GroupByColsIndexMap {
    pub page_start: usize,
    pub page_end: usize,
    pub sorted_group_by_start: usize,
    pub sorted_group_by_end: usize,
    pub aggregated: usize,
    pub partial_aggregated: usize,
    pub is_final: usize,
    pub eq_next: usize,
    pub is_less_than_tuple_aux_start: usize,
    pub is_less_than_tuple_aux_end: usize,
    pub is_equal_vec_aux_start: usize,
    pub is_equal_vec_aux_end: usize,
}

impl<T: Clone> GroupByCols<T> {
    pub fn from_slice(slc: &[T], group_by_air: &GroupByAir) -> Self {
        let index_map = GroupByCols::<T>::index_map(&group_by_air);

        let page = slc[index_map.page_start..index_map.page_end].to_vec();
        let sorted_group_by =
            slc[index_map.sorted_group_by_start..index_map.sorted_group_by_end].to_vec();
        let aggregated = slc[index_map.aggregated].clone();
        let partial_aggregated = slc[index_map.partial_aggregated].clone();
        let is_final = slc[index_map.is_final].clone();
        let eq_next = slc[index_map.eq_next].clone();
        let is_less_than_tuple_aux = IsLessThanTupleAuxCols::from_slice(
            &slc[index_map.is_less_than_tuple_aux_start..index_map.is_less_than_tuple_aux_end],
            group_by_air.is_less_than_tuple_air.limb_bits(),
            group_by_air.is_less_than_tuple_air.decomp(),
            group_by_air.group_by_cols.len(),
        );
        let is_equal_vec_aux = IsEqualVecAuxCols::from_slice(
            &slc[index_map.is_equal_vec_aux_start..index_map.is_equal_vec_aux_end],
            group_by_air.group_by_cols.len(),
        );

        Self {
            page,
            sorted_group_by,
            aggregated,
            partial_aggregated,
            is_final,
            eq_next,
            is_less_than_tuple_aux,
            is_equal_vec_aux,
        }
    }

    pub fn index_map(group_by_air: &GroupByAir) -> GroupByColsIndexMap {
        let num_group_by = group_by_air.group_by_cols.len();
        let lt_tuple_width = IsLessThanTupleAuxCols::<T>::get_width(
            group_by_air.is_less_than_tuple_air.limb_bits(),
            group_by_air.is_less_than_tuple_air.decomp(),
            num_group_by,
        );
        let eq_vec_width = IsEqualVecAuxCols::<T>::get_width(num_group_by);

        let page_idxs = (0, group_by_air.page_width);
        let sorted_group_by_idxs = (page_idxs.1, page_idxs.1 + num_group_by);
        let aggregated_idx = sorted_group_by_idxs.1;
        let partial_aggregated_idx = aggregated_idx + 1;
        let is_final_idx = partial_aggregated_idx + 1;
        let eq_next_idx = is_final_idx + 1;
        let is_less_than_tuple_aux_idxs = (eq_next_idx + 1, eq_next_idx + 1 + lt_tuple_width);
        let is_equal_vec_aux_idxs = (
            is_less_than_tuple_aux_idxs.1,
            is_less_than_tuple_aux_idxs.1 + eq_vec_width,
        );

        GroupByColsIndexMap {
            page_start: page_idxs.0,
            page_end: page_idxs.1,
            sorted_group_by_start: sorted_group_by_idxs.0,
            sorted_group_by_end: sorted_group_by_idxs.1,
            aggregated: aggregated_idx,
            partial_aggregated: partial_aggregated_idx,
            is_final: is_final_idx,
            eq_next: eq_next_idx,
            is_less_than_tuple_aux_start: is_less_than_tuple_aux_idxs.0,
            is_less_than_tuple_aux_end: is_less_than_tuple_aux_idxs.1,
            is_equal_vec_aux_start: is_equal_vec_aux_idxs.0,
            is_equal_vec_aux_end: is_equal_vec_aux_idxs.1,
        }
    }

    pub fn get_width(group_by_air: &GroupByAir) -> usize {
        let index_map = GroupByCols::<T>::index_map(&group_by_air);
        index_map.is_equal_vec_aux_end
    }
}
