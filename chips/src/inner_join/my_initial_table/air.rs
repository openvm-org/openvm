use afs_stark_backend::air_builders::PartitionedAirBuilder;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use super::{columns::TableCols, MyInitialTableAir};
use crate::sub_chip::AirConfig;

impl<F: Field> BaseAir<F> for MyInitialTableAir {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl AirConfig for MyInitialTableAir {
    type Cols<T> = TableCols<T>;
}

impl<AB: PartitionedAirBuilder> Air<AB> for MyInitialTableAir
where
    AB::M: Clone,
{
    fn eval(&self, builder: &mut AB) {
        let table_trace: &<AB as AirBuilder>::M = &builder.partitioned_main()[0].clone();
        let aux_trace: &<AB as AirBuilder>::M = &builder.partitioned_main()[1].clone();

        let (table_local, aux_local) = (table_trace.row_slice(0), aux_trace.row_slice(0));

        let is_alloc = table_local[0];
        let mult_cnt = aux_local[0];

        // Ensuring that mult_cnt is zero if is_alloc is zero
        // This is important because we never want to send/receive data if
        // the row in unallocated
        builder.assert_eq(mult_cnt, mult_cnt * is_alloc);
    }
}
