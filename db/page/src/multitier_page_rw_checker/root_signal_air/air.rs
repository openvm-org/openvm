use afs_primitives::sub_chip::AirConfig;
use afs_stark_backend::{
    air_builders::PartitionedAirBuilder,
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use super::{columns::RootSignalCols, RootSignalAir};

impl<F: Field, const COMMITMENT_LEN: usize> BaseAir<F> for RootSignalAir<COMMITMENT_LEN> {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl<F: Field, const COMMITMENT_LEN: usize> PartitionedBaseAir<F>
    for RootSignalAir<COMMITMENT_LEN>
{
}

impl<F: Field, const COMMITMENT_LEN: usize> BaseAirWithPublicValues<F>
    for RootSignalAir<COMMITMENT_LEN>
{
    fn num_public_values(&self) -> usize {
        COMMITMENT_LEN
    }
}

impl<const COMMITMENT_LEN: usize> AirConfig for RootSignalAir<COMMITMENT_LEN> {
    type Cols<T> = RootSignalCols<T>;
}

impl<
        AB: AirBuilder + AirBuilderWithPublicValues + PartitionedAirBuilder + InteractionBuilder,
        const COMMITMENT_LEN: usize,
    > Air<AB> for RootSignalAir<COMMITMENT_LEN>
where
    AB::M: Clone,
{
    fn eval(&self, builder: &mut AB) {
        // only constrain that root_commitment is accurate according to public values
        let main: &<AB as AirBuilder>::M = &builder.common_main().clone();
        let local = main.row_slice(0);
        let pi = builder.public_values().to_vec();
        self.eval_interactions(
            builder,
            &RootSignalCols::from_slice(&local, self.idx_len, self.is_init),
            &pi,
        );
    }
}
