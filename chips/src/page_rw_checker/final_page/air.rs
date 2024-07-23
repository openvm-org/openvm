use afs_stark_backend::air_builders::PartitionedAirBuilder;
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::columns::IndexedPageWriteAuxCols;
use super::{columns::IndexedPageWriteCols, IndexedPageWriteAir};
use crate::indexed_output_page_air::columns::IndexedOutputPageCols;
use crate::sub_chip::AirConfig;
use crate::{common::page_cols::PageCols, sub_chip::SubAir};

impl<F: Field> BaseAir<F> for IndexedPageWriteAir {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl AirConfig for IndexedPageWriteAir {
    type Cols<T> = IndexedPageWriteCols<T>;
}

impl<AB: PartitionedAirBuilder + InteractionBuilder> Air<AB> for IndexedPageWriteAir
where
    AB::M: Clone,
{
    fn eval(&self, builder: &mut AB) {
        let io = [0, 1].map(|i| {
            PageCols::from_slice(
                &builder.partitioned_main()[0].row_slice(i),
                self.final_air.idx_len,
                self.final_air.data_len,
            )
        });
        let aux = [0, 1].map(|i| {
            IndexedPageWriteAuxCols::from_slice(&builder.partitioned_main()[1].row_slice(i), self)
        });
        // Making sure the page is in the proper format
        SubAir::eval(self, builder, io, aux);
    }
}

impl<AB: PartitionedAirBuilder + InteractionBuilder> SubAir<AB> for IndexedPageWriteAir {
    type IoView = [PageCols<AB::Var>; 2];
    type AuxView = [IndexedPageWriteAuxCols<AB::Var>; 2];

    fn eval(&self, builder: &mut AB, io: Self::IoView, aux: Self::AuxView) {
        // Making sure the page is in the proper format

        // Ensuring that rcv_mult is always 1 or 3 times is_alloc (ensures it's always 0, 1, or 3) on the next row (which has net effect of being on every row)
        let local_is_alloc = io[1].is_alloc;
        let local_rcv_mult = aux[1].rcv_mult;
        self.eval_interactions(
            builder,
            &IndexedPageWriteCols {
                final_page_cols: IndexedOutputPageCols {
                    page_cols: io[0].clone(),
                    aux_cols: aux[0].final_page_aux_cols.clone(),
                },
                rcv_mult: aux[0].rcv_mult,
            },
        );
        SubAir::eval(
            &self.final_air,
            builder,
            io,
            aux[1].final_page_aux_cols.clone(),
        );
        builder.assert_zero(
            (local_rcv_mult - local_is_alloc)
                * (local_rcv_mult - AB::Expr::from_canonical_u8(3) * local_is_alloc),
        );
    }
}
