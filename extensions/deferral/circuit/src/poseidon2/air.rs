use std::{borrow::Borrow, iter::once, sync::Arc};

use openvm_circuit_primitives::{AlignedBorrow, ColumnsAir};
use openvm_poseidon2_air::{
    Poseidon2Config, Poseidon2SubAir, Poseidon2SubCols, BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS,
};
use openvm_stark_backend::{
    air_builders::sub::SubAirBuilder,
    interaction::{InteractionBuilder, LookupBus},
    p3_air::{Air, AirBuilder, BaseAir},
    p3_matrix::Matrix,
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
use p3_field::{Field, PrimeCharacteristicRing};

use super::SBOX_REGISTERS;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct DeferralPoseidon2Cols<T> {
    pub inner: Poseidon2SubCols<T, SBOX_REGISTERS>,
    pub compress_mult: T,
    pub capacity_mult: T,
}

pub struct DeferralPoseidon2Air<F: Field> {
    pub subair: Arc<Poseidon2SubAir<F, SBOX_REGISTERS>>,
    pub bus: LookupBus,
}

impl<F: Field> ColumnsAir for DeferralPoseidon2Air<F> {}

impl<F: Field> DeferralPoseidon2Air<F> {
    pub fn new(config: Poseidon2Config<F>, bus: LookupBus) -> Self {
        Self {
            subair: Arc::new(Poseidon2SubAir::new(config.constants.into())),
            bus,
        }
    }
}

impl<F: Field> BaseAir<F> for DeferralPoseidon2Air<F> {
    fn width(&self) -> usize {
        DeferralPoseidon2Cols::<F>::width()
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for DeferralPoseidon2Air<F> {}
impl<F: Field> PartitionedBaseAir<F> for DeferralPoseidon2Air<F> {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for DeferralPoseidon2Air<AB::F> {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main
            .row_slice(0)
            .expect("window should have at least one row");
        let local: &DeferralPoseidon2Cols<AB::Var> = (*local).borrow();

        let mut sub_builder =
            SubAirBuilder::<AB, Poseidon2SubAir<AB::F, SBOX_REGISTERS>, AB::F>::new(
                builder,
                0..self.subair.width(),
            );
        self.subair.eval(&mut sub_builder);

        let inputs = local.inner.inputs;
        let compress_res = &local.inner.ending_full_rounds
            [BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS - 1]
            .post[..DIGEST_SIZE];
        let capacity_res = &local.inner.ending_full_rounds
            [BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS - 1]
            .post[DIGEST_SIZE..];

        self.bus.add_key_with_lookups(
            builder,
            inputs
                .into_iter()
                .map(Into::into)
                .chain(compress_res.iter().copied().map(Into::into))
                .chain(once(AB::Expr::ONE)),
            local.compress_mult,
        );

        self.bus.add_key_with_lookups(
            builder,
            inputs
                .into_iter()
                .map(Into::into)
                .chain(capacity_res.iter().copied().map(Into::into))
                .chain(once(AB::Expr::ZERO)),
            local.capacity_mult,
        );
    }
}
