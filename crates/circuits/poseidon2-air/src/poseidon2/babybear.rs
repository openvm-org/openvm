use std::array::from_fn;

use lazy_static::lazy_static;
use openvm_stark_backend::p3_field::AbstractField;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use zkhash::{
    ark_ff::PrimeField as _, fields::babybear::FpBabyBear as HorizenBabyBear,
    poseidon2::poseidon2_instance_babybear::RC16,
};

use super::{
    Poseidon2Constants, POSEIDON2_HALF_FULL_ROUNDS, POSEIDON2_PARTIAL_ROUNDS, POSEIDON2_WIDTH,
};

pub(crate) fn horizen_to_p3_babybear(horizen_babybear: HorizenBabyBear) -> BabyBear {
    BabyBear::from_canonical_u64(horizen_babybear.into_bigint().0[0])
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
    let p_end = POSEIDON2_HALF_FULL_ROUNDS + POSEIDON2_PARTIAL_ROUNDS;

    let beginning_full_round_constants: [[BabyBear; POSEIDON2_WIDTH]; POSEIDON2_HALF_FULL_ROUNDS] =
        from_fn(|i| p3_rc16[i].clone().try_into().unwrap());
    let partial_round_constants: [BabyBear; POSEIDON2_PARTIAL_ROUNDS] =
        from_fn(|i| p3_rc16[i + POSEIDON2_HALF_FULL_ROUNDS][0]);
    let ending_full_round_constants: [[BabyBear; POSEIDON2_WIDTH]; POSEIDON2_HALF_FULL_ROUNDS] =
        from_fn(|i| p3_rc16[i + p_end].clone().try_into().unwrap());

    Poseidon2Constants {
        beginning_full_round_constants,
        partial_round_constants,
        ending_full_round_constants,
    }
}

lazy_static! {
    pub static ref BABYBEAR_BEGIN_EXT_CONSTS: [[BabyBear; POSEIDON2_WIDTH]; POSEIDON2_HALF_FULL_ROUNDS] =
        horizen_round_consts().beginning_full_round_constants;
    pub static ref BABYBEAR_PARTIAL_CONSTS: [BabyBear; POSEIDON2_PARTIAL_ROUNDS] =
        horizen_round_consts().partial_round_constants;
    pub static ref BABYBEAR_END_EXT_CONSTS: [[BabyBear; POSEIDON2_WIDTH]; POSEIDON2_HALF_FULL_ROUNDS] =
        horizen_round_consts().ending_full_round_constants;
}

pub(crate) fn babybear_internal_linear_layer<F: AbstractField, const WIDTH: usize>(
    state: &mut [F; WIDTH],
    int_diag_m1_matrix: [F; 16],
    reduction_factor: F,
) {
    let sum = state.iter().cloned().sum::<F>();
    for (input, diag_m1) in state.iter_mut().zip(int_diag_m1_matrix) {
        *input = (sum.clone() + diag_m1 * input.clone()) * reduction_factor.clone();
    }
}

// Same as p3_poseidon2::mds_light_permutation when for WIDTH = POSEIDON2_WIDTH
pub(crate) fn babybear_external_linear_layer<F: AbstractField, const WIDTH: usize>(
    input: &mut [F; WIDTH],
    ext_mds_matrix: [[F; 4]; 4],
) {
    let mut new_state: [F; WIDTH] = core::array::from_fn(|_| F::ZERO);
    for i in (0..WIDTH).step_by(4) {
        for index1 in 0..4 {
            for index2 in 0..4 {
                new_state[i + index1] +=
                    ext_mds_matrix[index1][index2].clone() * input[i + index2].clone();
            }
        }
    }

    let sums: [F; 4] = core::array::from_fn(|j| {
        (0..WIDTH)
            .step_by(4)
            .map(|i| new_state[i + j].clone())
            .sum()
    });

    for i in 0..WIDTH {
        new_state[i] += sums[i % 4].clone();
    }

    input.clone_from_slice(&new_state);
}
