use super::GroupByAir;
use crate::is_equal_vec::columns::IsEqualVecAuxCols;
use std::ops::Range;

/// Since `GroupByChip` contains a `LessThanChip` subchip and an `IsEqualVecChip` subchip, a subset of
/// the columns are those of the `LessThanChip` and `IsEqualVecChip`.
///
/// The `io` columns consist only of the cached page, because output is sent to `MyFinalPage`. The
/// `aux` columns are all other columns.
///
/// Implements two methods:
///
/// * `from_slice`: Takes a slice and returns a `GroupByCols` struct.
/// * `index_map`: Returns a `GroupByColsIndexMap` struct, used to index all other structs and
///   defines the order of segments in a slice.
pub struct GroupByCols<T> {
    pub io: GroupByIOCols<T>,
    pub aux: GroupByAuxCols<T>,
}

/// The `io` columns consist only of the cached page, because output is sent to `MyFinalPage`.
pub struct GroupByIOCols<T> {
    pub is_allocated: T,
    pub page: Vec<T>,
}

/// The `aux` columns are all non-cached columns.
pub struct GroupByAuxCols<T> {
    pub sorted_group_by_alloc: T,
    pub sorted_group_by: Vec<T>,
    pub sorted_group_by_combined: Vec<T>,
    pub aggregated: T,
    pub partial_aggregated: T,
    pub is_final: T,
    pub eq_next: T,
    pub is_equal_vec_aux: IsEqualVecAuxCols<T>,
}

/// Maps parts of the `GroupByCols` to their indices. Note that `sorted_group_by_combined_range` is
/// a range containing `sorted_group_by_alloc` and `sorted_group_by_range`.
pub struct GroupByColsIndexMap {
    pub allocated_idx: usize,
    pub page_range: Range<usize>,
    pub sorted_group_by_alloc: usize,
    pub sorted_group_by_range: Range<usize>,
    pub sorted_group_by_combined_range: Range<usize>,
    pub aggregated: usize,
    pub partial_aggregated: usize,
    pub is_final: usize,
    pub eq_next: usize,
    pub is_equal_vec_aux_range: Range<usize>,
}

impl<T: Clone> GroupByCols<T> {
    /// Takes a slice and returns a `GroupByCols` struct.
    pub fn from_slice(slc: &[T], group_by_air: &GroupByAir) -> Self {
        assert!(slc.len() == group_by_air.get_width());
        let index_map = GroupByCols::<T>::index_map(group_by_air);

        let is_allocated = slc[index_map.allocated_idx].clone();
        let page = slc[index_map.page_range].to_vec();
        let sorted_group_by_alloc = slc[index_map.sorted_group_by_alloc].clone();
        let sorted_group_by = slc[index_map.sorted_group_by_range].to_vec();
        let sorted_group_by_combined = slc[index_map.sorted_group_by_combined_range].to_vec();
        let aggregated = slc[index_map.aggregated].clone();
        let partial_aggregated = slc[index_map.partial_aggregated].clone();
        let is_final = slc[index_map.is_final].clone();
        let eq_next = slc[index_map.eq_next].clone();
        let is_equal_vec_aux = IsEqualVecAuxCols::from_slice(
            &slc[index_map.is_equal_vec_aux_range],
            group_by_air.group_by_cols.len() + 1,
        );

        Self {
            io: GroupByIOCols { is_allocated, page },
            aux: GroupByAuxCols {
                sorted_group_by_alloc,
                sorted_group_by,
                sorted_group_by_combined,
                aggregated,
                partial_aggregated,
                is_final,
                eq_next,
                is_equal_vec_aux,
            },
        }
    }

    /// Returns a `GroupByColsIndexMap` struct, used to index all other structs and defines the
    /// order of segments in a slice.
    pub fn index_map(group_by_air: &GroupByAir) -> GroupByColsIndexMap {
        let num_group_by = group_by_air.group_by_cols.len();
        let eq_vec_width = IsEqualVecAuxCols::<T>::get_width(num_group_by + 1);

        let (allocated_idx, page_range, sorted_group_by_alloc) = if !group_by_air.sorted {
            let allocated_idx = 0;
            let page_range = allocated_idx + 1..group_by_air.page_width;
            let sorted_group_by_alloc = page_range.end;
            (allocated_idx, page_range, sorted_group_by_alloc)
        } else {
            let allocated_idx = 0;
            let page_range = 0..0;
            let sorted_group_by_alloc = 0;
            (allocated_idx, page_range, sorted_group_by_alloc)
        };
        let sorted_group_by_range =
            sorted_group_by_alloc + 1..sorted_group_by_alloc + 1 + num_group_by;
        let sorted_group_by_combined_range = sorted_group_by_alloc..sorted_group_by_range.end;
        let aggregated_idx = sorted_group_by_range.end;
        let partial_aggregated_idx = aggregated_idx + 1;
        let is_final_idx = partial_aggregated_idx + 1;
        let eq_next_idx = is_final_idx + 1;
        let is_equal_vec_aux_range = eq_next_idx + 1..eq_next_idx + 1 + eq_vec_width;

        GroupByColsIndexMap {
            allocated_idx,
            page_range,
            sorted_group_by_alloc,
            sorted_group_by_range,
            sorted_group_by_combined_range,
            aggregated: aggregated_idx,
            partial_aggregated: partial_aggregated_idx,
            is_final: is_final_idx,
            eq_next: eq_next_idx,
            is_equal_vec_aux_range,
        }
    }
}
