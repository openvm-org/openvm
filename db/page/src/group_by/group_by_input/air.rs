use std::{collections::HashMap, iter};

use afs_primitives::{
    is_equal_vec::{columns::IsEqualVecIoCols, IsEqualVecAir},
    sub_chip::{AirConfig, SubAir},
};
use afs_stark_backend::{air_builders::PartitionedAirBuilder, interaction::InteractionBuilder};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use super::{
    columns::{GroupByAuxCols, GroupByCols, GroupedPageCols},
    GroupByOperation,
};
use crate::common::{page::Page, page_cols::PageCols};

impl<F: Field> BaseAir<F> for GroupByAir {
    fn width(&self) -> usize {
        self.get_width()
    }
}

/// Main struct defining constraints and dimensions for group-by operation
///
/// Operation:
/// 1. sends columns of interest to itself, constraining equal rows to be adjacent
/// 2. completes partial operations on aggregated column
/// 3. sends the aggregated columns to MyFinalPage
pub struct GroupByAir {
    pub internal_bus: usize,
    pub output_bus: usize,

    /// Has +1 to check equality on `is_alloc` column
    pub is_equal_vec_air: IsEqualVecAir,

    /// Includes is_allocated column, so `data_len + 1 == page_width`
    pub page_width: usize,
    pub group_by_cols: Vec<usize>,
    pub aggregated_col: usize,

    /// Whether the input page is already sorted by the group-by columns
    pub sorted: bool,

    /// The operation to perform on the aggregated column
    pub op: GroupByOperation,
}

impl GroupByAir {
    pub fn new(
        page_width: usize,
        group_by_cols: Vec<usize>,
        aggregated_col: usize,
        internal_bus: usize,
        output_bus: usize,
        sorted: bool,
        op: GroupByOperation,
    ) -> Self {
        Self {
            page_width,
            // has +1 to check equality on is_alloc column
            is_equal_vec_air: IsEqualVecAir::new(group_by_cols.len() + 1),
            group_by_cols,
            aggregated_col,
            sorted,
            op,
            internal_bus,
            output_bus,
        }
    }

    /// Width of entire trace
    pub fn get_width(&self) -> usize {
        if !self.sorted {
            self.page_width + 3 * self.group_by_cols.len() + 6
        } else {
            3 * self.group_by_cols.len() + 6
        }
    }

    /// Width of auxilliary trace, i.e. all non-input-page columns
    pub fn aux_width(&self) -> usize {
        if !self.sorted {
            3 * self.group_by_cols.len() + 6
        } else {
            2 * self.group_by_cols.len() + 4
        }
    }

    pub fn select_and_sort(&self, page: &Page) -> Vec<Vec<u32>> {
        if self.sorted {
            page.iter()
                .filter(|row| row.is_alloc == 1)
                .map(|row| row.data.clone())
                .collect()
        } else {
            let mut grouped_page: Vec<Vec<u32>> = page
                .iter()
                .filter(|row| row.is_alloc == 1)
                .map(|row| {
                    let mut selected_row: Vec<u32> = self
                        .group_by_cols
                        .iter()
                        .map(|&col_index| row.data[col_index])
                        .collect();
                    selected_row.push(row.data[self.aggregated_col]);
                    selected_row
                })
                .collect();
            grouped_page.sort();
            grouped_page
        }
    }

    /// This pure function computes the answer to the group-by operation
    pub fn request(&self, page: &Page) -> (Page, Page) {
        let grouped_page: Vec<Vec<u32>> = self.select_and_sort(page);

        let mut sums_by_key: HashMap<Vec<u32>, u32> = HashMap::new();
        for row in grouped_page.iter() {
            let (value, index) = row.split_last().unwrap();
            *sums_by_key.entry(index.to_vec()).or_insert(0) += value;
        }

        // Convert the hashmap back to a sorted vector for further processing
        let mut grouped_sums: Vec<Vec<u32>> = sums_by_key
            .into_iter()
            .map(|(mut key, sum)| {
                key.insert(0, 1);
                key.push(sum);
                key
            })
            .collect();
        grouped_sums.sort();

        let idx_len = self.group_by_cols.len();
        let row_width = 1 + idx_len + 1;
        grouped_sums.resize(page.height(), vec![0; row_width]);

        let mut new_grouped_page: Vec<Vec<u32>> = grouped_page
            .iter()
            .map(|row| {
                let mut new_row = vec![1];
                new_row.append(&mut row.clone());
                new_row
            })
            .collect();
        new_grouped_page.resize(page.height(), vec![0; row_width]);
        (
            Page::from_2d_vec(&grouped_sums, idx_len, 1),
            Page::from_2d_vec(&new_grouped_page, idx_len, 1),
        )
    }
}

impl<AB: InteractionBuilder + PartitionedAirBuilder> Air<AB> for GroupByAir
where
    AB::M: Clone,
{
    /// Re-references builder into page_trace and aux_trace, then slices into local and next rows
    /// to evaluate using SubAir::eval(GroupByAir)
    fn eval(&self, builder: &mut AB) {
        let page_trace: &<AB as AirBuilder>::M = &builder.partitioned_main()[0];
        let aux_trace: &<AB as AirBuilder>::M = &builder.partitioned_main()[1];

        // get the current row and the next row
        let (local_page, next_page) = (page_trace.row_slice(0), page_trace.row_slice(1));
        let (local_aux, next_aux) = (aux_trace.row_slice(0), aux_trace.row_slice(1));

        let local_cols = GroupByCols::from_partitioned_slice(&local_page, &local_aux, self);
        let next_cols = GroupByCols::from_partitioned_slice(&next_page, &next_aux, self);
        drop((local_page, next_page, local_aux, next_aux));

        SubAir::eval(
            self,
            builder,
            (local_cols.page, next_cols.page),
            (local_cols.aux, next_cols.aux),
        );
    }
}

impl AirConfig for GroupByAir {
    type Cols<T> = GroupByCols<T>;
}

impl<AB: InteractionBuilder + PartitionedAirBuilder> SubAir<AB> for GroupByAir {
    /// `io` consists of `(local_page, next_page)` only the page, including `is_alloc`
    type IoView = (PageCols<AB::Var>, PageCols<AB::Var>);
    /// `aux.0` is `local.aux`, `aux.1` is `next.aux`.
    ///
    /// `aux` consists of everything that isn't `io`, including
    /// `sorted_group_by`, `sorted_group_by_alloc`, `aggregated`, and `partial_aggregated`
    type AuxView = (GroupByAuxCols<AB::Var>, GroupByAuxCols<AB::Var>);

    /// Constrains `sorted_group_by` along with `partial_aggregated` to hold correct values
    /// with minimal constraints.
    ///
    /// In fact `sorted_group_by` is not necessarily sorted. The only constraints are that
    /// allocated rows are placed at the beginning, and like rows are placed together.
    ///
    /// Like rows being placed together is enforced by the constraints on `MyFinalPage`, since
    /// all rows marked `final` are sent to MyFinalPage and hence must be pairwise distinct.
    fn eval(
        &self,
        builder: &mut AB,
        (local_page, next_page): Self::IoView,
        (local_aux, next_aux): Self::AuxView,
    ) {
        self.eval_interactions(
            builder,
            GroupByCols {
                page: local_page.clone(),
                aux: local_aux.clone(),
            },
        );

        // Return either grouped columns (if not None) or the respective columns from the page
        // (the latter occurs if the page is already assumed to be grouped appropriated)
        let get_grouped = |page: PageCols<AB::Var>, grouped: Option<GroupedPageCols<AB::Var>>| {
            grouped.unwrap_or_else(|| GroupedPageCols {
                is_alloc: page.is_alloc,
                group_by: self.group_by_cols.iter().map(|&i| page.data[i]).collect(),
                to_aggregate: page.data[self.aggregated_col],
            })
        };
        let local_grouped = get_grouped(local_page, local_aux.grouped);
        let next_grouped = get_grouped(next_page, next_aux.grouped);

        let is_equal_io = IsEqualVecIoCols {
            x: iter::once(local_grouped.is_alloc)
                .chain(local_grouped.group_by)
                .collect(),
            y: iter::once(next_grouped.is_alloc)
                .chain(next_grouped.group_by)
                .collect(),
            is_equal: local_aux.eq_next,
        };

        // constrain eq_next to hold the correct value
        SubAir::eval(
            &self.is_equal_vec_air,
            &mut builder.when_transition(),
            is_equal_io,
            local_aux.is_equal_vec_aux,
        );

        // if grouped is_alloc changes, then is_final must be 1, even if eq_next is also 1
        let grouped_is_alloc_diff = local_grouped.is_alloc - next_grouped.is_alloc;
        builder.when_transition().assert_eq(
            grouped_is_alloc_diff.clone(),
            grouped_is_alloc_diff * local_aux.is_final,
        );

        // constrain is_final to be 1 iff eq_next == false and local_grouped.is_alloc is 1
        builder.assert_eq(
            local_aux.is_final,
            local_grouped.is_alloc - local_grouped.is_alloc * local_aux.eq_next,
        );

        // constrain last vector equality to 0
        // because previously only constrained on transition
        builder.when_last_row().assert_zero(local_aux.eq_next);

        // initialize partial sum at first row
        builder
            .when_first_row()
            .assert_eq(local_aux.partial_aggregated, local_grouped.to_aggregate);

        // constrain partials to sum correctly
        builder.when_transition().assert_eq(
            next_aux.partial_aggregated,
            local_aux.eq_next * local_aux.partial_aggregated + next_grouped.to_aggregate,
        );

        // constrain allocated rows come first
        builder.when_transition().assert_eq(
            next_grouped.is_alloc * local_grouped.is_alloc,
            next_grouped.is_alloc,
        );
    }
}
