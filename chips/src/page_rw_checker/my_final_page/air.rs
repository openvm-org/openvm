use afs_stark_backend::air_builders::PartitionedAirBuilder;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::columns::MyFinalPageCols;
use super::{columns::MyFinalPageAuxCols, MyFinalPageAir};
use crate::sub_chip::AirConfig;
use crate::{common::page_cols::PageCols, sub_chip::SubAir};

impl<F: Field> BaseAir<F> for MyFinalPageAir {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl AirConfig for MyFinalPageAir {
    type Cols<T> = MyFinalPageCols<T>;
}

impl<AB: PartitionedAirBuilder> Air<AB> for MyFinalPageAir
where
    AB::M: Clone,
{
    fn eval(&self, builder: &mut AB) {
        let page_trace: &<AB as AirBuilder>::M = &builder.partitioned_main()[0].clone();
        let aux_trace: &<AB as AirBuilder>::M = &builder.partitioned_main()[1].clone();

        let page_local = page_trace.row_slice(0);
        let aux_local = aux_trace.row_slice(0);

        // Making sure the page is in the proper format
        Air::eval(&self.final_air, builder);

        // Ensuring that rcv_mult is always 1 or 3 times is_alloc (ensures it's always 0, 1, or 3)
        let local_is_alloc = page_local[0];
        let local_rcv_mult = aux_local[aux_local.len() - 1];
        builder.assert_zero(
            (local_rcv_mult - local_is_alloc)
                * (local_rcv_mult - AB::Expr::from_canonical_u8(3) * local_is_alloc),
        );
    }
}

impl<AB: AirBuilder + PartitionedAirBuilder> SubAir<AB> for MyFinalPageAir
where
    AB::M: Clone,
{
    type IoView = [PageCols<AB::Var>; 2];
    type AuxView = MyFinalPageAuxCols<AB::Var>;

    fn eval(&self, builder: &mut AB, io: Self::IoView, aux: Self::AuxView) {
        // Making sure the page is in the proper format

        // Ensuring that rcv_mult is always 1 or 3 times is_alloc (ensures it's always 0, 1, or 3) on the next row (which has net effect of being on every row)
        let local_is_alloc = io[1].is_alloc;
        let local_rcv_mult = aux.rcv_mult;
        SubAir::eval(&self.final_air, builder, io, aux.final_page_aux_cols);
        builder.assert_zero(
            (local_rcv_mult - local_is_alloc)
                * (local_rcv_mult - AB::Expr::from_canonical_u8(3) * local_is_alloc),
        );
    }
}
