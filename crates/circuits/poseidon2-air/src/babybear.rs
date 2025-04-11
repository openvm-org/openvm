use std::{any::TypeId, array::from_fn};

use lazy_static::lazy_static;
use openvm_stark_backend::p3_field::{Field, FieldAlgebra, PrimeField32};
use openvm_stark_sdk::p3_baby_bear::{BabyBear, BabyBearInternalLayerParameters};
use p3_monty_31::InternalLayerBaseParameters;
use p3_poseidon2::{mds_light_permutation, GenericPoseidon2LinearLayers, MDSMat4};
use zkhash::{
    ark_ff::PrimeField as _, fields::babybear::FpBabyBear as HorizenBabyBear,
    poseidon2::poseidon2_instance_babybear::RC16,
};

use super::{Poseidon2Constants, POSEIDON2_HALF_FULL_ROUNDS, POSEIDON2_WIDTH};

pub const BABYBEAR_POSEIDON2_PARTIAL_ROUNDS: usize = 13;
pub const BABYBEAR_POSEIDON2_SBOX_DEGREE: u64 = 7;
// pub const BABYBEAR_POSEIDON2_SBOX_REGISTERS: usize = 1;

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
    let p_end = POSEIDON2_HALF_FULL_ROUNDS + BABYBEAR_POSEIDON2_PARTIAL_ROUNDS;

    let beginning_full_round_constants: [[BabyBear; POSEIDON2_WIDTH]; POSEIDON2_HALF_FULL_ROUNDS] =
        from_fn(|i| p3_rc16[i].clone().try_into().unwrap());
    let partial_round_constants: [BabyBear; BABYBEAR_POSEIDON2_PARTIAL_ROUNDS] =
        from_fn(|i| p3_rc16[i + POSEIDON2_HALF_FULL_ROUNDS][0]);
    let ending_full_round_constants: [[BabyBear; POSEIDON2_WIDTH]; POSEIDON2_HALF_FULL_ROUNDS] =
        from_fn(|i| p3_rc16[i + p_end].clone().try_into().unwrap());

    Poseidon2Constants {
        beginning_full_round_constants,
        partial_round_constants: partial_round_constants.to_vec(),
        ending_full_round_constants,
    }
}

lazy_static! {
    static ref BABYBEAR_CONSTS: Poseidon2Constants<BabyBear> = horizen_round_consts();
    pub static ref BABYBEAR_BEGIN_EXT_CONSTS: [[BabyBear; POSEIDON2_WIDTH]; POSEIDON2_HALF_FULL_ROUNDS] =
        BABYBEAR_CONSTS.beginning_full_round_constants;
    pub static ref BABYBEAR_PARTIAL_CONSTS: Vec<BabyBear> =
        BABYBEAR_CONSTS.partial_round_constants.clone();
    pub static ref BABYBEAR_END_EXT_CONSTS: [[BabyBear; POSEIDON2_WIDTH]; POSEIDON2_HALF_FULL_ROUNDS] =
        BABYBEAR_CONSTS.ending_full_round_constants;
}

pub(crate) fn babybear_internal_linear_layer<FA: FieldAlgebra, const WIDTH: usize>(
    state: &mut [FA; WIDTH],
    int_diag_m1_matrix: &[FA::F; WIDTH],
) {
    let sum = state.iter().cloned().sum::<FA>();
    for (input, diag_m1) in state.iter_mut().zip(int_diag_m1_matrix) {
        *input = sum.clone() + FA::from_f(*diag_m1) * input.clone();
    }
}

// Default round constants for only BabyBear, but we convert to `F` due to some annoyances with
// generics. This should only be used concretely when `F = BabyBear`.
pub fn default_baby_bear_rc<F: Field>() -> Poseidon2Constants<F> {
    let convert_field = |f: BabyBear| F::from_canonical_u32(f.as_canonical_u32());
    Poseidon2Constants {
        beginning_full_round_constants: BABYBEAR_BEGIN_EXT_CONSTS.map(|x| x.map(convert_field)),
        partial_round_constants: BABYBEAR_PARTIAL_CONSTS
            .iter()
            .cloned()
            .map(convert_field)
            .collect(),
        ending_full_round_constants: BABYBEAR_END_EXT_CONSTS.map(|x| x.map(convert_field)),
    }
}

/// This type needs to implement GenericPoseidon2LinearLayers generic in F so that our Poseidon2SubAir can also
/// be generic in F, but in reality each implementation of this struct's functions should be field specific. To
/// circumvent this, Poseidon2LinearLayers is generic in F but **currently requires** that F is BabyBear.
#[derive(Debug, Clone)]
pub struct BabyBearPoseidon2LinearLayers;

// This is the same as the implementation for GenericPoseidon2LinearLayersMonty31<BabyBearParameters, BabyBearInternalLayerParameters> except that we drop the
// clause that FA needs be multipliable by BabyBear.
// TODO[jpw/stephen]: This is clearly not the best way to do this, but it would
// require some reworking in plonky3 to get around the generics.
impl<FA: FieldAlgebra> GenericPoseidon2LinearLayers<FA, POSEIDON2_WIDTH>
    for BabyBearPoseidon2LinearLayers
{
    fn internal_linear_layer(state: &mut [FA; POSEIDON2_WIDTH]) {
        let diag_m1_matrix = &<BabyBearInternalLayerParameters as InternalLayerBaseParameters<
            _,
            16,
        >>::INTERNAL_DIAG_MONTY;
        assert_eq!(
            TypeId::of::<FA::F>(),
            TypeId::of::<BabyBear>(),
            "BabyBear is the only supported field type"
        );
        let diag_m1_matrix = unsafe {
            std::mem::transmute::<&[BabyBear; POSEIDON2_WIDTH], &[FA::F; POSEIDON2_WIDTH]>(
                diag_m1_matrix,
            )
        };
        babybear_internal_linear_layer(state, diag_m1_matrix);
    }

    fn external_linear_layer(state: &mut [FA; POSEIDON2_WIDTH]) {
        mds_light_permutation(state, &MDSMat4);
    }
}
