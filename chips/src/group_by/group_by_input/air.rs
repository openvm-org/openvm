use std::iter;

use afs_stark_backend::air_builders::PartitionedAirBuilder;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use crate::common::page_cols::PageCols;
use crate::is_equal_vec::columns::IsEqualVecIOCols;
use crate::sub_chip::{AirConfig, SubAir};

use super::columns::{GroupByAuxCols, GroupByCols, GroupedPageCols};
use super::GroupByAir;

impl<F: Field> BaseAir<F> for GroupByAir {
    fn width(&self) -> usize {
        self.get_width()
    }
}

impl<AB: PartitionedAirBuilder> Air<AB> for GroupByAir
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

impl<AB: PartitionedAirBuilder> SubAir<AB> for GroupByAir {
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

        let is_equal_io = IsEqualVecIOCols {
            x: iter::once(local_grouped.is_alloc)
                .chain(local_grouped.group_by)
                .collect(),
            y: iter::once(next_grouped.is_alloc)
                .chain(next_grouped.group_by)
                .collect(),
            prod: local_aux.eq_next,
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
