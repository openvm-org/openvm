use std::sync::LazyLock;

use halo2_base::utils::ScalarField as _;
use itertools::Itertools;
pub(crate) use openvm_stark_sdk::config::baby_bear_bn254_poseidon2::WIDTH as POSEIDON2_WIDTH;
use zkhash::{
    ark_ff::{BigInteger, PrimeField as _},
    fields::bn256::FpBN256 as ark_FpBN256,
    poseidon2::poseidon2_instance_bn256::{MAT_DIAG3_M_1, RC3},
};

use crate::Fr;

pub mod poseidon2;
use poseidon2::Poseidon2Params;

/// Poseidon2-BN256 round counts from zkhash (HorizenLabs/poseidon2, rev bb476b9).
const ROUNDS_F: usize = 8;
const ROUNDS_P: usize = 56;

fn convert_fr(input: ark_FpBN256) -> Fr {
    Fr::from_bytes_le(&input.into_bigint().to_bytes_le())
}

pub(crate) static POSEIDON2_PARAMS: LazyLock<Poseidon2Params<Fr, POSEIDON2_WIDTH>> =
    LazyLock::new(|| {
        let mut round_constants: Vec<[Fr; POSEIDON2_WIDTH]> = RC3
            .iter()
            .map(|vec| {
                vec.iter()
                    .cloned()
                    .map(convert_fr)
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
            .collect();

        let rounds_f_beginning = ROUNDS_F / 2;
        let p_end = rounds_f_beginning + ROUNDS_P;
        let internal_round_constants = round_constants
            .drain(rounds_f_beginning..p_end)
            .map(|vec| vec[0])
            .collect::<Vec<_>>();
        let external_round_constants = round_constants;
        Poseidon2Params::new(
            ROUNDS_F,
            ROUNDS_P,
            MAT_DIAG3_M_1
                .iter()
                .copied()
                .map(convert_fr)
                .collect_vec()
                .try_into()
                .unwrap(),
            external_round_constants,
            internal_round_constants,
        )
    });
