use core::array;
use std::sync::LazyLock;

use halo2_base::utils::biguint_to_fe;
pub(crate) use openvm_stark_sdk::config::baby_bear_bn254_poseidon2::SPONGE_WIDTH as POSEIDON2_WIDTH;
use openvm_stark_sdk::{
    config::bn254_poseidon2::{
        default_bn254_poseidon2_width2_constants, default_bn254_poseidon2_width3_constants,
        Poseidon2Bn254Constants,
    },
    openvm_stark_backend::p3_field::PrimeField,
};

use crate::Fr;

pub mod poseidon2;
use poseidon2::Poseidon2Params;

/// Width-2 compression permutation width.
const COMPRESS_WIDTH: usize = 2;

/// Convert `Poseidon2Bn254Constants` from stark-backend into halo2 `Poseidon2Params`.
fn bn254_constants_to_params<const WIDTH: usize>(
    constants: &Poseidon2Bn254Constants<WIDTH>,
) -> Poseidon2Params<Fr, WIDTH> {
    use openvm_stark_sdk::config::baby_bear_bn254_poseidon2::Bn254Scalar;

    let bn254_to_fr = |elem: &Bn254Scalar| -> Fr { biguint_to_fe(&elem.as_canonical_biguint()) };

    let initial_ext: Vec<[Fr; WIDTH]> = constants
        .initial_external_rc()
        .iter()
        .map(|rc| array::from_fn(|i| bn254_to_fr(&rc[i])))
        .collect();
    let terminal_ext: Vec<[Fr; WIDTH]> = constants
        .terminal_external_rc()
        .iter()
        .map(|rc| array::from_fn(|i| bn254_to_fr(&rc[i])))
        .collect();
    let internal_rc: Vec<Fr> = constants.internal_rc().iter().map(bn254_to_fr).collect();
    let mat_internal_diag_m_1 =
        array::from_fn(|i| bn254_to_fr(&constants.mat_internal_diag_m_1()[i]));

    let mut external_rc = initial_ext;
    external_rc.extend(terminal_ext);

    let rounds_f = external_rc.len();
    let rounds_p = internal_rc.len();
    Poseidon2Params::new(
        rounds_f,
        rounds_p,
        mat_internal_diag_m_1,
        external_rc,
        internal_rc,
    )
}

/// Width-3 Poseidon2 params for leaf hashing and transcript sponge.
pub(crate) static POSEIDON2_PARAMS: LazyLock<Poseidon2Params<Fr, POSEIDON2_WIDTH>> =
    LazyLock::new(|| bn254_constants_to_params(default_bn254_poseidon2_width3_constants()));

/// Width-2 Poseidon2 params for Merkle compression (gnark-crypto compatible).
pub(crate) static POSEIDON2_COMPRESS_PARAMS: LazyLock<Poseidon2Params<Fr, COMPRESS_WIDTH>> =
    LazyLock::new(|| bn254_constants_to_params(default_bn254_poseidon2_width2_constants()));
