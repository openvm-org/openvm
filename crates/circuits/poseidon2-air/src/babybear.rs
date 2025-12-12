use std::array::from_fn;

use lazy_static::lazy_static;
use openvm_stark_backend::p3_field::{
    integers::QuotientMap, PrimeCharacteristicRing, PrimeField32,
};
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use zkhash::{
    ark_ff::PrimeField as _, fields::babybear::FpBabyBear as HorizenBabyBear,
    poseidon2::poseidon2_instance_babybear::RC16,
};

use super::{
    Poseidon2Constants, BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS, BABY_BEAR_POSEIDON2_PARTIAL_ROUNDS,
    POSEIDON2_WIDTH,
};

pub(crate) fn horizen_to_p3_babybear(horizen_babybear: HorizenBabyBear) -> BabyBear {
    BabyBear::from_u64(horizen_babybear.into_bigint().0[0])
}

pub(crate) fn horizen_round_consts() -> Poseidon2Constants<BabyBear> {
    let p3_rc16: Vec<Vec<BabyBear>> = RC16
        .iter()
        .map(|round| {
            round
                .iter()
                .map(|babybear| horizen_to_p3_babybear(*babybear))
                .collect()
        })
        .collect();
    let p_end = BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS + BABY_BEAR_POSEIDON2_PARTIAL_ROUNDS;

    let beginning_full_round_constants: [[BabyBear; POSEIDON2_WIDTH];
        BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS] = from_fn(|i| p3_rc16[i].clone().try_into().unwrap());
    let partial_round_constants: [BabyBear; BABY_BEAR_POSEIDON2_PARTIAL_ROUNDS] =
        from_fn(|i| p3_rc16[i + BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS][0]);
    let ending_full_round_constants: [[BabyBear; POSEIDON2_WIDTH];
        BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS] =
        from_fn(|i| p3_rc16[i + p_end].clone().try_into().unwrap());

    Poseidon2Constants {
        beginning_full_round_constants,
        partial_round_constants,
        ending_full_round_constants,
    }
}

lazy_static! {
    pub static ref BABYBEAR_BEGIN_EXT_CONSTS: [[BabyBear; POSEIDON2_WIDTH]; BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS] =
        horizen_round_consts().beginning_full_round_constants;
    pub static ref BABYBEAR_PARTIAL_CONSTS: [BabyBear; BABY_BEAR_POSEIDON2_PARTIAL_ROUNDS] =
        horizen_round_consts().partial_round_constants;
    pub static ref BABYBEAR_END_EXT_CONSTS: [[BabyBear; POSEIDON2_WIDTH]; BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS] =
        horizen_round_consts().ending_full_round_constants;
}

/// The vector `[-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/2^27, -1/2^8, -1/16,
/// -1/2^27]` saved as an array of BabyBear elements. Copied from the previous Poseidon2
/// implementation to preserve the exact constraint structure.
pub const INTERNAL_DIAG_MONTY_16: [BabyBear; 16] = BabyBear::new_array([
    BabyBear::ORDER_U32 - 2,
    1,
    2,
    (BabyBear::ORDER_U32 + 1) >> 1,
    3,
    4,
    (BabyBear::ORDER_U32 - 1) >> 1,
    BabyBear::ORDER_U32 - 3,
    BabyBear::ORDER_U32 - 4,
    BabyBear::ORDER_U32 - ((BabyBear::ORDER_U32 - 1) >> 8),
    BabyBear::ORDER_U32 - ((BabyBear::ORDER_U32 - 1) >> 2),
    BabyBear::ORDER_U32 - ((BabyBear::ORDER_U32 - 1) >> 3),
    BabyBear::ORDER_U32 - 15,
    (BabyBear::ORDER_U32 - 1) >> 8,
    (BabyBear::ORDER_U32 - 1) >> 4,
    15,
]);

pub(crate) fn babybear_internal_linear_layer<R: PrimeCharacteristicRing>(
    state: &mut [R; POSEIDON2_WIDTH],
    diag_m1_matrix: &[BabyBear; POSEIDON2_WIDTH],
) {
    let sum: R = state.iter().cloned().sum();
    for (val, &diag_elem) in state.iter_mut().zip(diag_m1_matrix.iter()) {
        let diag_sub = R::PrimeSubfield::from_int(diag_elem.as_canonical_u32());
        let diag_r = R::from_prime_subfield(diag_sub);
        *val = sum.clone() + diag_r * val.clone();
    }
}
