use std::borrow::Borrow;

use afs_stark_backend::air_builders::PartitionedAirBuilder;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use crate::is_equal_vec::columns::{IsEqualVecCols, IsEqualVecIOCols};
use crate::sub_chip::{AirConfig, SubAir};

use super::columns::{GroupByAuxCols, GroupByCols, GroupByIOCols};
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
        let page_trace: &<AB as AirBuilder>::M = &builder.partitioned_main()[0].clone();
        let aux_trace: &<AB as AirBuilder>::M = &builder.partitioned_main()[1].clone();

        // get the current row and the next row
        let (local_page, next_page) = (page_trace.row_slice(0), page_trace.row_slice(1));
        let (local_aux, next_aux) = (aux_trace.row_slice(0), aux_trace.row_slice(1));
        let local_page: &[AB::Var] = (*local_page).borrow();
        let next_page: &[AB::Var] = (*next_page).borrow();
        let local_aux: &[AB::Var] = (*local_aux).borrow();
        let next_aux: &[AB::Var] = (*next_aux).borrow();

        let local: &[AB::Var] = &[local_page, local_aux].concat();
        let next: &[AB::Var] = &[next_page, next_aux].concat();

        if local.len() != next.len() {
            panic!("Local and next lengths do not match.");
        }
        if local.len() != self.get_width() {
            panic!("Local length does not match the expected width.");
        }

        let local_cols = GroupByCols::<AB::Var>::from_slice(local, self);

        let next_cols = GroupByCols::<AB::Var>::from_slice(next, self);

        SubAir::eval(
            self,
            builder,
            (local_cols.io, next_cols.io),
            (local_cols.aux, next_cols.aux),
        );
    }
}

impl AirConfig for GroupByAir {
    type Cols<T> = GroupByCols<T>;
}

impl<AB: PartitionedAirBuilder> SubAir<AB> for GroupByAir {
    /// `io.0` is `local.io`, `io.1` is `next.io`.
    ///
    /// `io` consists of only the page, including `is_alloc`
    type IoView = (GroupByIOCols<AB::Var>, GroupByIOCols<AB::Var>);
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
    fn eval(&self, builder: &mut AB, _io: Self::IoView, aux: Self::AuxView) {
        let is_equal_vec_cols = IsEqualVecCols {
            io: IsEqualVecIOCols {
                x: aux.0.sorted_group_by_combined,
                y: aux.1.sorted_group_by_combined,
                prod: aux.0.eq_next,
            },
            aux: aux.0.is_equal_vec_aux,
        };

        // constrain eq_next to hold the correct value
        SubAir::eval(
            &self.is_equal_vec_air,
            &mut builder.when_transition(),
            is_equal_vec_cols.io,
            is_equal_vec_cols.aux,
        );

        // if sorted_group_by_alloc changes, then is_final must be 1, even if eq_next is 1
        builder.when_transition().assert_eq(
            aux.0.sorted_group_by_alloc - aux.1.sorted_group_by_alloc,
            (aux.0.sorted_group_by_alloc - aux.1.sorted_group_by_alloc) * aux.0.is_final,
        );

        // constrain is_final to be 1 iff differ from next row and sorted_group_by_alloc is 1
        builder.assert_eq(
            aux.0.is_final,
            aux.0.sorted_group_by_alloc - aux.0.sorted_group_by_alloc * aux.0.eq_next,
        );

        // constrain last vector equality to 0
        // because previously only constrained on transition
        builder.when_last_row().assert_zero(aux.0.eq_next);

        // initialize partial sum at first row
        builder
            .when_first_row()
            .assert_eq(aux.0.partial_aggregated, aux.0.aggregated);

        // constrain partials to sum correctly
        builder.when_transition().assert_eq(
            aux.1.partial_aggregated,
            aux.0.eq_next * aux.0.partial_aggregated + aux.1.aggregated,
        );

        // constrain allocated rows come first
        builder.when_transition().assert_eq(
            aux.1.sorted_group_by_alloc * aux.0.sorted_group_by_alloc,
            aux.1.sorted_group_by_alloc,
        );
    }
}
