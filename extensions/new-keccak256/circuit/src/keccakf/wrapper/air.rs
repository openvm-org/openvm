
use openvm_stark_backend::interaction::PermutationCheckBus;
use openvm_stark_backend::rap::BaseAirWithPublicValues;
use openvm_stark_backend::rap::PartitionedBaseAir;
use openvm_stark_backend::p3_air::{Air, BaseAir};
use openvm_stark_backend::interaction::InteractionBuilder;
use p3_keccak_air::KeccakAir;
use crate::keccakf::wrapper::columns::KeccakfWrapperCols;
use crate::keccakf::wrapper::columns::NUM_KECCAKF_WRAPPER_COLS;
use openvm_stark_backend::p3_matrix::Matrix;
use core::borrow::{Borrow, BorrowMut};
use std::iter::once;

#[derive(derive_new::new)]
pub struct KeccakfWrapperAir {
    pub keccak_bus: PermutationCheckBus
}

impl<F> BaseAirWithPublicValues<F> for KeccakfWrapperAir {}
impl<F> PartitionedBaseAir<F> for KeccakfWrapperAir {}
impl<F> BaseAir<F> for KeccakfWrapperAir {
    fn width(&self) -> usize {
        NUM_KECCAKF_WRAPPER_COLS
    }
}

impl<AB: InteractionBuilder> Air<AB> for KeccakfWrapperAir {
    fn eval(&self, builder: &mut AB) {
        let keccak_f_air = KeccakAir {};
        keccak_f_air.eval(builder);

        let main = builder.main();
        let local = main.row_slice(0);
        let local: &KeccakfWrapperCols<AB::Var> = (*local).borrow();

        // communicate with the keccakbus
        self.keccak_bus.receive(
            builder,
            local.inner.preimage.into_iter().flat_map(|y| y.into_iter()).flat_map(|x| x.into_iter()).map(Into::into).chain(once(local.request_id.into())),
            local.is_enabled
        );

        self.keccak_bus.receive(
            builder,
            local.inner.a_prime_prime.into_iter().flat_map(|y| y.into_iter()).flat_map(|x| x.into_iter()).map(Into::into).chain(once(local.request_id.into())),
            local.is_enabled
        );
    }
}