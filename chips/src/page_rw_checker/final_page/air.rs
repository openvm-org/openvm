use afs_stark_backend::air_builders::PartitionedAirBuilder;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::{columns::MyFinalPageCols, MyFinalPageAir};
use crate::{
    final_page::columns::FinalPageCols,
    sub_chip::{AirConfig, SubAir},
};

impl<F: Field> BaseAir<F> for MyFinalPageAir {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl AirConfig for MyFinalPageAir {
    type Cols<T> = FinalPageCols<T>;
}

impl<AB: PartitionedAirBuilder> Air<AB> for MyFinalPageAir
where
    AB::M: Clone,
{
    fn eval(&self, builder: &mut AB) {
        let page_trace: &<AB as AirBuilder>::M = &builder.partitioned_main()[0].clone();
        let aux_trace: &<AB as AirBuilder>::M = &builder.partitioned_main()[1].clone();

        let (page_local, page_next) = (page_trace.row_slice(0), page_trace.row_slice(1));
        let (aux_local, aux_next) = (aux_trace.row_slice(0), aux_trace.row_slice(1));

        let my_final_page_local_cols = MyFinalPageCols::from_slice(
            page_local
                .iter()
                .chain(aux_local[..aux_local.len()].iter())
                .copied()
                .collect::<Vec<_>>()
                .as_slice(),
            self.final_air.clone(),
        );

        let my_final_page_next_cols = MyFinalPageCols::from_slice(
            page_next
                .iter()
                .chain(aux_next[..aux_next.len()].iter())
                .copied()
                .collect::<Vec<_>>()
                .as_slice(),
            self.final_air.clone(),
        );

        // Ensuring that rcv_mult is always 1 or 3 times is_alloc (ensures it's always 0, 1, or 3)
        let local_is_alloc = my_final_page_local_cols.final_page_cols.page_cols.is_alloc;
        let local_rcv_mult = my_final_page_local_cols.rcv_mult;
        builder.assert_zero(
            (local_rcv_mult - local_is_alloc)
                * (local_rcv_mult - AB::Expr::from_canonical_u8(3) * local_is_alloc),
        );

        // Ensuring the page is in the proper format
        SubAir::eval(
            &self.final_air,
            builder,
            [
                my_final_page_local_cols.final_page_cols.page_cols,
                my_final_page_next_cols.final_page_cols.page_cols,
            ],
            my_final_page_next_cols.final_page_cols.aux_cols,
        );
    }
}
