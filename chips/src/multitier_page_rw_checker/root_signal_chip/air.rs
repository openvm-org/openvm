use afs_stark_backend::air_builders::PartitionedAirBuilder;
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use crate::{
    is_less_than_tuple::columns::IsLessThanTupleIOCols,
    page_rw_checker::page_chip::columns::PageCols,
    sub_chip::{AirConfig, SubAir},
};

use super::{columns::RootSignalCols, RootSignalChip};

impl<F: Field, const COMMITMENT_LEN: usize> BaseAir<F> for RootSignalChip<COMMITMENT_LEN> {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl<const COMMITMENT_LEN: usize> AirConfig for RootSignalChip<COMMITMENT_LEN> {
    type Cols<T> = RootSignalCols<T>;
}

impl<
        AB: AirBuilder + AirBuilderWithPublicValues + PartitionedAirBuilder,
        const COMMITMENT_LEN: usize,
    > Air<AB> for RootSignalChip<COMMITMENT_LEN>
where
    AB::M: Clone,
{
    fn eval(&self, builder: &mut AB) {
        // only constrain that own_commitment is accurate
        // partition is physical page data vs metadata
        let main: &<AB as AirBuilder>::M = &builder.partitioned_main()[0].clone();
        let local = main.row_slice(0);
        let pi = builder.public_values().to_vec();
        for i in 0..COMMITMENT_LEN {
            builder.assert_eq(pi[i], local[i]);
        }
    }
}
