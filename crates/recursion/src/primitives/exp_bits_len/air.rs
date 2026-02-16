use core::borrow::Borrow;

use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::F;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::primitives::bus::{ExpBitsLenBus, ExpBitsLenMessage};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct ExpBitsLenCols<T> {
    pub is_valid: T,
    pub base: T,
    pub bit_src: T,
    pub num_bits: T,
    pub num_bits_inv: T,
    pub result: T,
    pub sub_result: T,
    // bit_src = 2 * q_mod_2 + r_mod_2
    pub bit_src_div_2: T,
    pub bit_src_mod_2: T,
}

#[derive(Debug)]
pub struct ExpBitsLenAir {
    pub exp_bits_len_bus: ExpBitsLenBus,
}

impl ExpBitsLenAir {
    pub fn new(exp_bits_len_bus: ExpBitsLenBus) -> Self {
        Self { exp_bits_len_bus }
    }
}

impl BaseAirWithPublicValues<F> for ExpBitsLenAir {}
impl PartitionedBaseAir<F> for ExpBitsLenAir {}

impl<F> BaseAir<F> for ExpBitsLenAir {
    fn width(&self) -> usize {
        ExpBitsLenCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for ExpBitsLenAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0).expect("window should have two elements");
        let local: &ExpBitsLenCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.bit_src_mod_2);
        builder.assert_eq(
            local.bit_src,
            local.bit_src_div_2 * AB::Expr::TWO + local.bit_src_mod_2,
        );
        builder.assert_eq(
            local.num_bits,
            local.num_bits * local.num_bits * local.num_bits_inv,
        );
        builder.when(local.num_bits).assert_eq(
            local.result,
            local.sub_result
                * (local.bit_src_mod_2 * local.base + AB::Expr::ONE - local.bit_src_mod_2),
        );

        let is_num_bits_nonzero = local.num_bits * local.num_bits_inv;
        self.exp_bits_len_bus.add_key_with_lookups(
            builder,
            ExpBitsLenMessage {
                base: local.base,
                bit_src: local.bit_src,
                num_bits: local.num_bits,
                result: local.result,
            },
            local.is_valid,
        );
        self.exp_bits_len_bus.lookup_key(
            builder,
            ExpBitsLenMessage {
                base: local.base * local.base,
                bit_src: local.bit_src_div_2.into(),
                num_bits: local.num_bits - AB::Expr::ONE,
                result: local.sub_result.into(),
            },
            is_num_bits_nonzero.clone(),
        );
        builder
            .when(AB::Expr::ONE - is_num_bits_nonzero)
            .assert_one(local.result);
    }
}
