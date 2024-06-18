use afs_stark_backend::air_builders::PartitionedAirBuilder;
use p3_air::{Air, BaseAir};
use p3_field::Field;

use crate::sub_chip::AirConfig;

use super::{columns::PageIndexScanOutputCols, PageIndexScanOutputAir};

impl AirConfig for PageIndexScanOutputAir {
    type Cols<T> = PageIndexScanOutputCols<T>;
}

impl<F: Field> BaseAir<F> for PageIndexScanOutputAir {
    fn width(&self) -> usize {
        PageIndexScanOutputCols::<F>::get_width(&self.final_page_air)
    }
}

impl<AB: PartitionedAirBuilder> Air<AB> for PageIndexScanOutputAir
where
    AB::M: Clone,
{
    fn eval(&self, builder: &mut AB) {
        // Making sure the page is in the proper format
        Air::eval(&self.final_page_air, builder);
    }
}
