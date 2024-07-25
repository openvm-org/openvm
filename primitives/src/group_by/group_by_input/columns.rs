use super::GroupByAir;
use crate::{common::page_cols::PageCols, is_equal_vec::columns::IsEqualVecAuxCols};
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
    /// Page with `idx_len = 0` because the page does not need an index.
    pub page: PageCols<T>,
    pub aux: GroupByAuxCols<T>,
}

/// The `aux` columns are all non-cached columns.
#[derive(Clone)]
pub struct GroupByAuxCols<T> {
    /// If page is not already grouped, extra columns are needed to do the grouping.
    pub grouped: Option<GroupedPageCols<T>>,
    pub partial_aggregated: T,
    pub is_final: T,
    pub eq_next: T,
    pub is_equal_vec_aux: IsEqualVecAuxCols<T>,
}

/// The columns that are relevant to the GroupBy operation.
#[derive(Clone)]
pub struct GroupedPageCols<T> {
    pub is_alloc: T,
    pub group_by: Vec<T>,
    pub to_aggregate: T,
}

/// Maps parts of the `GroupByCols` to their indices. Note that `sorted_group_by_combined_range` is
/// a range containing `sorted_group_by_alloc` and `sorted_group_by_range`. Indexes by the
/// respective partition, not by the complete row.
pub struct GroupByColsIndexMap {
    pub allocated_idx: usize,
    pub page_range: Range<usize>,
    pub sorted_group_by_alloc: usize,
    pub sorted_group_by_range: Range<usize>,
    pub sorted_group_by_combined_range: Range<usize>,
    pub to_aggregate: usize,
    pub partial_aggregated: usize,
    pub is_final: usize,
    pub eq_next: usize,
    pub is_equal_vec_aux_range: Range<usize>,
}

impl<T: Clone> GroupByCols<T> {
    /// Takes a slice and returns a `GroupByCols` struct.
    pub fn from_slice(slc: &[T], group_by_air: &GroupByAir) -> Self {
        assert_eq!(slc.len(), group_by_air.get_width());
        Self::from_partitioned_slice(
            &slc[..group_by_air.page_width],
            &slc[group_by_air.page_width..],
            group_by_air,
        )
    }

    pub fn from_partitioned_slice(page: &[T], aux: &[T], group_by_air: &GroupByAir) -> Self {
        assert_eq!(page.len(), group_by_air.page_width);
        assert_eq!(
            aux.len(),
            group_by_air.get_width() - group_by_air.page_width
        );

        let index_map = GroupByCols::<T>::index_map(group_by_air);

        let grouped = if !group_by_air.sorted {
            Some(GroupedPageCols {
                is_alloc: aux[index_map.sorted_group_by_alloc].clone(),
                group_by: aux[index_map.sorted_group_by_range].to_vec(),
                to_aggregate: aux[index_map.to_aggregate].clone(),
            })
        } else {
            None
        };

        let partial_aggregated = aux[index_map.partial_aggregated].clone();
        let is_final = aux[index_map.is_final].clone();
        let eq_next = aux[index_map.eq_next].clone();
        let is_equal_vec_aux = IsEqualVecAuxCols::from_slice(
            &aux[index_map.is_equal_vec_aux_range],
            group_by_air.group_by_cols.len() + 1,
        );

        let data_len = group_by_air.page_width - 1;
        Self {
            page: PageCols::from_slice(page, 0, data_len),
            aux: GroupByAuxCols {
                grouped,
                partial_aggregated,
                is_final,
                eq_next,
                is_equal_vec_aux,
            },
        }
    }

    /// Returns a `GroupByColsIndexMap` struct, used to index all other structs and defines the
    /// order of segments in a slice. Indexes by the respective partition, not by the complete row.
    pub fn index_map(group_by_air: &GroupByAir) -> GroupByColsIndexMap {
        let num_group_by = group_by_air.group_by_cols.len();
        let eq_vec_width = IsEqualVecAuxCols::<T>::width(num_group_by + 1);

        let allocated_idx = 0;
        let page_range = if !group_by_air.sorted {
            allocated_idx + 1..group_by_air.page_width
        } else {
            0..0
        };
        let sorted_group_by_alloc = 0;
        let sorted_group_by_range =
            sorted_group_by_alloc + 1..sorted_group_by_alloc + 1 + num_group_by;
        let sorted_group_by_combined_range = sorted_group_by_alloc..sorted_group_by_range.end;
        let aggregated_idx = sorted_group_by_range.end;
        let partial_aggregated_idx = if !group_by_air.sorted {
            aggregated_idx + 1
        } else {
            0
        };
        let is_final_idx = partial_aggregated_idx + 1;
        let eq_next_idx = is_final_idx + 1;
        let is_equal_vec_aux_range = eq_next_idx + 1..eq_next_idx + 1 + eq_vec_width;

        GroupByColsIndexMap {
            allocated_idx,
            page_range,
            sorted_group_by_alloc,
            sorted_group_by_range,
            sorted_group_by_combined_range,
            to_aggregate: aggregated_idx,
            partial_aggregated: partial_aggregated_idx,
            is_final: is_final_idx,
            eq_next: eq_next_idx,
            is_equal_vec_aux_range,
        }
    }
}
