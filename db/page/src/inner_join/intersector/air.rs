use afs_primitives::{
    is_less_than_tuple::columns::IsLessThanTupleIoCols,
    sub_chip::{AirConfig, SubAir},
    utils::or,
};
use afs_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use super::{
    columns::{IntersectorAuxCols, IntersectorCols, IntersectorIoCols},
    IntersectorAir,
};

impl<F: Field> BaseAirWithPublicValues<F> for IntersectorAir {}
impl<F: Field> PartitionedBaseAir<F> for IntersectorAir {}
impl<F: Field> BaseAir<F> for IntersectorAir {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl AirConfig for IntersectorAir {
    type Cols<T> = IntersectorCols<T>;
}

impl<AB: InteractionBuilder> Air<AB> for IntersectorAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let [local, next] = [0, 1].map(|i| IntersectorCols::from_slice(&main.row_slice(i), self));

        SubAir::eval(self, builder, [local.io, next.io], [local.aux, next.aux]);
    }
}

impl<AB: InteractionBuilder> SubAir<AB> for IntersectorAir {
    type IoView = [IntersectorIoCols<AB::Var>; 2];
    type AuxView = [IntersectorAuxCols<AB::Var>; 2];

    /// Ensures indices in non-extra rows are sorted and distinct,
    /// that out_mult is correct, and that multiplicity are zero for
    /// non-extra rows
    fn eval(&self, builder: &mut AB, io: Self::IoView, aux: Self::AuxView) {
        let [local_io_cols, next_io_cols] = io;
        let [_, next_aux_cols] = aux;

        // Ensuring that rows are sorted by idx
        let lt_io_cols = IsLessThanTupleIoCols {
            x: local_io_cols.idx.clone(),
            y: next_io_cols.idx,
            tuple_less_than: next_aux_cols.lt_out,
        };

        self.lt_chip
            .eval_when_transition(builder, lt_io_cols, next_aux_cols.lt_aux);

        builder
            .when_transition()
            .assert_one(or(next_io_cols.is_extra, next_aux_cols.lt_out));

        // Ensuring out_mult is correct
        builder.assert_eq(
            local_io_cols.t1_mult * local_io_cols.t2_mult,
            local_io_cols.out_mult,
        );

        // Ensuting that t1_mult, t2_mult, out_mult are zeros when is_extra is one
        builder.assert_zero(local_io_cols.is_extra * local_io_cols.t1_mult);
        builder.assert_zero(local_io_cols.is_extra * local_io_cols.t2_mult);
        builder.assert_zero(local_io_cols.is_extra * local_io_cols.out_mult);

        self.eval_interactions(builder, local_io_cols);
    }
}
