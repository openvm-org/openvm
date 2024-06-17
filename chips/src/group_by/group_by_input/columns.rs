use super::GroupByAir;
use crate::is_equal_vec::columns::IsEqualVecAuxCols;
use std::ops::Range;

// Since GroupByChip contains a LessThanChip subchip and an IsEqualVecChip subchip, a subset of the
// columns are those of the LessThanChip and IsEqualVecChip
pub struct GroupByCols<T> {
    pub io: GroupByIOCols<T>,
    pub aux: GroupByAuxCols<T>,
}

pub struct GroupByIOCols<T> {
    pub is_allocated: T,
    pub page: Vec<T>,
}

pub struct GroupByAuxCols<T> {
    pub sorted_group_by: Vec<T>,
    pub aggregated: T,
    pub partial_aggregated: T,
    pub is_final: T,
    pub eq_next: T,
    pub is_equal_vec_aux: IsEqualVecAuxCols<T>,
}

pub struct GroupByColsIndexMap {
    pub allocated_idx: usize,
    pub page_range: Range<usize>,
    pub sorted_group_by_alloc: usize,
    pub sorted_group_by_range: Range<usize>,
    pub aggregated: usize,
    pub partial_aggregated: usize,
    pub is_final: usize,
    pub eq_next: usize,
    pub is_equal_vec_aux_range: Range<usize>,
}

impl<T: Clone> GroupByCols<T> {
    pub fn from_slice(slc: &[T], group_by_air: &GroupByAir) -> Self {
        assert!(slc.len() == group_by_air.get_width());
        let index_map = GroupByCols::<T>::index_map(group_by_air);

        let is_allocated = slc[index_map.allocated_idx].clone();
        let page = slc[index_map.page_range].to_vec();
        let sorted_group_by = slc[index_map.sorted_group_by_range].to_vec();
        let aggregated = slc[index_map.aggregated].clone();
        let partial_aggregated = slc[index_map.partial_aggregated].clone();
        let is_final = slc[index_map.is_final].clone();
        let eq_next = slc[index_map.eq_next].clone();
        let is_equal_vec_aux = IsEqualVecAuxCols::from_slice(
            &slc[index_map.is_equal_vec_aux_range],
            group_by_air.group_by_cols.len(),
        );

        Self {
            io: GroupByIOCols { is_allocated, page },
            aux: GroupByAuxCols {
                sorted_group_by,
                aggregated,
                partial_aggregated,
                is_final,
                eq_next,
                is_equal_vec_aux,
            },
        }
    }

    pub fn index_map(group_by_air: &GroupByAir) -> GroupByColsIndexMap {
        let num_group_by = group_by_air.group_by_cols.len();
        let eq_vec_width = IsEqualVecAuxCols::<T>::get_width(num_group_by);

        let allocated_idx = 0;
        let page_range = allocated_idx + 1..group_by_air.page_width;
        let sorted_group_by_alloc = page_range.end;
        let sorted_group_by_range =
            sorted_group_by_alloc + 1..sorted_group_by_alloc + 1 + num_group_by;
        let aggregated_idx = sorted_group_by_range.end;
        let partial_aggregated_idx = aggregated_idx + 1;
        let is_final_idx = partial_aggregated_idx + 1;
        let eq_next_idx = is_final_idx + 1;
        let is_equal_vec_aux_range = eq_next_idx + 1..eq_next_idx + 1 + eq_vec_width;

        // TODO replace with Range
        // let page_range = Range::new(page_idxs.0, page_idxs.1);

        GroupByColsIndexMap {
            allocated_idx,
            page_range,
            sorted_group_by_alloc,
            sorted_group_by_range,
            aggregated: aggregated_idx,
            partial_aggregated: partial_aggregated_idx,
            is_final: is_final_idx,
            eq_next: eq_next_idx,
            is_equal_vec_aux_range,
        }
    }

    pub fn get_width(group_by_air: &GroupByAir) -> usize {
        let index_map = GroupByCols::<T>::index_map(group_by_air);
        index_map.is_equal_vec_aux_range.end
    }
}

impl<T> GroupByIOCols<T> {
    pub fn get_width(&self) -> usize {
        self.page.len() + 1
    }
}

impl<T: Clone> GroupByAuxCols<T> {
    pub fn get_width(&self) -> usize {
        self.sorted_group_by.len() + 4 + self.is_equal_vec_aux.get_self_width()
    }
}
