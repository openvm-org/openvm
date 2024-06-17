use afs_stark_backend::air_builders::PartitionedAirBuilder;
use p3_air::{Air, BaseAir};
use p3_field::Field;

use super::MyFinalPageAir;
use crate::{final_page::columns::FinalPageCols, sub_chip::AirConfig};

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
        Air::eval(&self.final_air, builder);
    }
}
