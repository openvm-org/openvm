use std::any::TypeId;

use lazy_static::lazy_static;
use once_cell::sync::Lazy;
use openvm_stark_backend::p3_field::{Field, FieldAlgebra, PrimeField32};
use openvm_stark_sdk::{
    config::{
        fri_params::SecurityParameters,
        koala_bear_poseidon2::{engine_from_perm, perm_from_constants, KoalaBearPoseidon2Engine},
        log_up_params::log_up_security_params_baby_bear_100_bits,
        FriParameters,
    },
    p3_koala_bear::{KoalaBear, KoalaBearInternalLayerParameters},
};
use p3_monty_31::InternalLayerBaseParameters;
use p3_poseidon2::{mds_light_permutation, GenericPoseidon2LinearLayers, MDSMat4};
use rand::distributions::Standard;
use rand::Rng;
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

use crate::{Poseidon2Constants, POSEIDON2_HALF_FULL_ROUNDS, POSEIDON2_WIDTH};

// KoalaBear only needs SBOX_DEGREE of 3, so we can always set SBOX_REGISTERS to 0.
pub const KOALABEAR_POSEIDON2_PARTIAL_ROUNDS: usize = 20;
pub const KOALABEAR_POSEIDON2_SBOX_DEGREE: u64 = 3;
pub const KOALABEAR_POSEIDON2_SBOX_REGISTERS: usize = 0;

pub(crate) fn rng_round_consts() -> Poseidon2Constants<KoalaBear> {
    let mut rng = ChaCha20Rng::from_seed(Default::default());
    let partial_round_constants: [KoalaBear; KOALABEAR_POSEIDON2_PARTIAL_ROUNDS] =
        core::array::from_fn(|_| rng.sample(Standard));

    Poseidon2Constants {
        beginning_full_round_constants: core::array::from_fn(|_| rng.sample(Standard)),
        partial_round_constants: partial_round_constants.to_vec(),
        ending_full_round_constants: core::array::from_fn(|_| rng.sample(Standard)),
    }
}

static KOALABEAR_CONSTS: Lazy<Poseidon2Constants<KoalaBear>> = Lazy::new(rng_round_consts);
lazy_static! {
    pub static ref KOALABEAR_BEGIN_EXT_CONSTS: [[KoalaBear; POSEIDON2_WIDTH]; POSEIDON2_HALF_FULL_ROUNDS] =
        KOALABEAR_CONSTS.beginning_full_round_constants;
    pub static ref KOALABEAR_PARTIAL_CONSTS: Vec<KoalaBear> =
        KOALABEAR_CONSTS.partial_round_constants.clone();
    pub static ref KOALABEAR_END_EXT_CONSTS: [[KoalaBear; POSEIDON2_WIDTH]; POSEIDON2_HALF_FULL_ROUNDS] =
        KOALABEAR_CONSTS.ending_full_round_constants;
}

pub fn default_koalabear_rc<F: Field>() -> Poseidon2Constants<F> {
    let convert_field = |f: KoalaBear| F::from_canonical_u32(f.as_canonical_u32());
    Poseidon2Constants {
        beginning_full_round_constants: KOALABEAR_BEGIN_EXT_CONSTS.map(|x| x.map(convert_field)),
        partial_round_constants: KOALABEAR_PARTIAL_CONSTS
            .iter()
            .cloned()
            .map(convert_field)
            .collect(),
        ending_full_round_constants: KOALABEAR_END_EXT_CONSTS.map(|x| x.map(convert_field)),
    }
}

#[derive(Debug, Clone)]
pub struct KoalaBearPoseidon2LinearLayers;

pub(crate) fn koalabear_internal_linear_layer<FA: FieldAlgebra, const WIDTH: usize>(
    state: &mut [FA; WIDTH],
    int_diag_m1_matrix: &[FA::F; WIDTH],
) {
    let sum = state.iter().cloned().sum::<FA>();
    for (input, diag_m1) in state.iter_mut().zip(int_diag_m1_matrix) {
        *input = sum.clone() + FA::from_f(*diag_m1) * input.clone();
    }
}

impl<FA: FieldAlgebra> GenericPoseidon2LinearLayers<FA, POSEIDON2_WIDTH>
    for KoalaBearPoseidon2LinearLayers
{
    fn internal_linear_layer(state: &mut [FA; POSEIDON2_WIDTH]) {
        let diag_m1_matrix = &<KoalaBearInternalLayerParameters as InternalLayerBaseParameters<
            _,
            16,
        >>::INTERNAL_DIAG_MONTY;
        assert_eq!(
            TypeId::of::<FA::F>(),
            TypeId::of::<KoalaBear>(),
            "KoalaBear is the only supported field type"
        );
        let diag_m1_matrix = unsafe {
            std::mem::transmute::<&[KoalaBear; POSEIDON2_WIDTH], &[FA::F; POSEIDON2_WIDTH]>(
                diag_m1_matrix,
            )
        };
        koalabear_internal_linear_layer(state, diag_m1_matrix);
    }

    fn external_linear_layer(state: &mut [FA; POSEIDON2_WIDTH]) {
        mds_light_permutation(state, &MDSMat4);
    }
}

pub fn koalabear_engine(fri_params: FriParameters) -> KoalaBearPoseidon2Engine {
    let perm = perm_from_constants::<POSEIDON2_WIDTH>(
        KOALABEAR_BEGIN_EXT_CONSTS.to_vec(),
        KOALABEAR_END_EXT_CONSTS.to_vec(),
        KOALABEAR_PARTIAL_CONSTS.clone(),
    );
    let security_params = SecurityParameters {
        fri_params,
        log_up_params: log_up_security_params_baby_bear_100_bits(),
    };
    engine_from_perm(perm, security_params)
}
