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
        GroupByCols::<F>::get_width(self)
    }
}

impl<AB: PartitionedAirBuilder> Air<AB> for GroupByAir
where
    AB::M: Clone,
{
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
    // io.0 is local.io, io.1 is next.io
    type IoView = (GroupByIOCols<AB::Var>, GroupByIOCols<AB::Var>);
    // aux.0 is local.aux, aux.1 is next.aux
    type AuxView = (GroupByAuxCols<AB::Var>, GroupByAuxCols<AB::Var>);

    fn eval(&self, builder: &mut AB, _io: Self::IoView, aux: Self::AuxView) {
        let is_equal_vec_cols = IsEqualVecCols {
            io: IsEqualVecIOCols {
                x: aux.0.sorted_group_by,
                y: aux.1.sorted_group_by,
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
        builder.when_last_row().assert_zero(aux.0.eq_next);

        builder.assert_one(aux.0.eq_next + aux.0.is_final);

        builder.when_transition().assert_eq(
            aux.1.aggregated,
            aux.0.eq_next * aux.0.aggregated + aux.0.aggregated,
        );
    }
}
