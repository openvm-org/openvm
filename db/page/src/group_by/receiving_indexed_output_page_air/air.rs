use afs_primitives::sub_chip::AirConfig;
use afs_stark_backend::{air_builders::PartitionedAirBuilder, interaction::InteractionBuilder};
use p3_air::{Air, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use super::ReceivingIndexedOutputPageAir;
use crate::common::page_cols::PageCols;
use crate::indexed_output_page_air::columns::IndexedOutputPageCols;

impl<F: Field> BaseAir<F> for ReceivingIndexedOutputPageAir {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl AirConfig for ReceivingIndexedOutputPageAir {
    type Cols<T> = IndexedOutputPageCols<T>;
}

impl<AB: PartitionedAirBuilder + InteractionBuilder> Air<AB> for ReceivingIndexedOutputPageAir
where
    <AB as p3_air::AirBuilder>::M: Clone,
{
    fn eval(&self, builder: &mut AB) {
        let page_trace: &<AB>::M = &builder.partitioned_main()[0];

        let page_local = PageCols::<AB::Var>::from_slice(
            &page_trace.row_slice(0),
            self.final_air.idx_len,
            self.final_air.data_len,
        );

        Air::eval(&self.final_air, builder);
        self.eval_interactions(builder, page_local);
    }
}
