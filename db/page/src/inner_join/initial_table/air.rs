use afs_stark_backend::{
    air_builders::PartitionedAirBuilder,
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use super::{columns::TableCols, InitialTableAir};

impl<F: Field> BaseAirWithPublicValues<F> for InitialTableAir {}
impl<F: Field> PartitionedBaseAir<F> for InitialTableAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![self.table_width()]
    }
    fn common_main_width(&self) -> usize {
        self.aux_width()
    }
}
impl<F: Field> BaseAir<F> for InitialTableAir {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl<AB: PartitionedAirBuilder + InteractionBuilder> Air<AB> for InitialTableAir {
    fn eval(&self, builder: &mut AB) {
        let page: &<AB as AirBuilder>::M = &builder.partitioned_main()[0];
        let aux: &<AB as AirBuilder>::M = &builder.partitioned_main()[1];

        let table_local = TableCols::from_partitioned_slice(
            &page.row_slice(0),
            &aux.row_slice(0),
            self.idx_len,
            self.data_len,
        );

        let is_alloc = table_local.page_cols.is_alloc;
        let mult_cnt = table_local.out_mult;

        // Ensuring that mult_cnt is zero if is_alloc is zero
        // This is important because we never want to send/receive data if
        // the row in unallocated
        builder.assert_eq(mult_cnt, mult_cnt * is_alloc);

        self.eval_interactions(builder, table_local.page_cols, mult_cnt);
    }
}
