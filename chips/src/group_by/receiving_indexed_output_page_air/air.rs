use afs_stark_backend::air_builders::PartitionedAirBuilder;
use p3_air::{Air, BaseAir};
use p3_field::Field;

use super::ReceivingIndexedOutputPageAir;
use crate::{indexed_output_page_air::columns::IndexedOutputPageCols, sub_chip::AirConfig};

impl<F: Field> BaseAir<F> for ReceivingIndexedOutputPageAir {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl AirConfig for ReceivingIndexedOutputPageAir {
    type Cols<T> = IndexedOutputPageCols<T>;
}

impl<AB: PartitionedAirBuilder> Air<AB> for ReceivingIndexedOutputPageAir
where
    AB::M: Clone,
{
    fn eval(&self, builder: &mut AB) {
        Air::eval(&self.final_air, builder);
    }
}
