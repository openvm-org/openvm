use std::sync::Arc;

use openvm_circuit_primitives::ColumnsAir;
use openvm_stark_backend::{
    keygen::types::MultiStarkVerifyingKey, p3_matrix::Matrix, AirRef, PartitionedBaseAir,
    StarkEngine, SystemParams,
};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, BaseAirWithPublicValues};

use super::DeferralCircuitPvs;
use crate::SC;

/// Minimal AIR with a single zero column and deferral-circuit public values.
///
/// This is used only as a cheap verifying-key seed for deferral path fixed-point derivation; it is
/// not intended to be proven.
pub(crate) struct EmptyAirWithPvs(pub(crate) usize);

// No columns provided: this dummy AIR has a single anonymous column and no matching `Cols` struct.
impl ColumnsAir for EmptyAirWithPvs {}

impl<F> BaseAir<F> for EmptyAirWithPvs {
    fn width(&self) -> usize {
        1
    }
}

impl<F> BaseAirWithPublicValues<F> for EmptyAirWithPvs {
    fn num_public_values(&self) -> usize {
        self.0
    }
}

impl<F> PartitionedBaseAir<F> for EmptyAirWithPvs {}

impl<AB: AirBuilder + AirBuilderWithPublicValues> Air<AB> for EmptyAirWithPvs {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).unwrap();
        builder.assert_zero(local[0].clone());

        let pvs = builder
            .public_values()
            .iter()
            .map(|pv| (*pv).into())
            .collect::<Vec<AB::Expr>>();
        for pv in pvs {
            builder.assert_eq(pv.clone(), pv);
        }
    }
}

pub fn dummy_deferral_circuit_vk<E>(system_params: SystemParams) -> Arc<MultiStarkVerifyingKey<SC>>
where
    E: StarkEngine<SC = SC>,
{
    let engine = E::new(system_params);
    let dummy_air = Arc::new(EmptyAirWithPvs(DeferralCircuitPvs::<u8>::width())) as AirRef<SC>;
    Arc::new(engine.keygen(&[dummy_air]).1)
}
