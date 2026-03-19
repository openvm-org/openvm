use std::sync::LazyLock;

use halo2_base::utils::ScalarField as _;
use itertools::Itertools;

use crate::Fr;

pub mod poseidon2;
use poseidon2::Poseidon2Params;

pub(crate) const POSEIDON2_WIDTH: usize = 3;
pub(crate) static POSEIDON2_PARAMS: LazyLock<Poseidon2Params<Fr, POSEIDON2_WIDTH>> =
    LazyLock::new(|| {
        use zkhash::{
            ark_ff::{BigInteger, PrimeField as _},
            fields::bn256::FpBN256 as ark_FpBN256,
            poseidon2::poseidon2_instance_bn256::{MAT_DIAG3_M_1, RC3},
        };

        fn convert_fr(input: ark_FpBN256) -> Fr {
            Fr::from_bytes_le(&input.into_bigint().to_bytes_le())
        }
        const T: usize = 3;
        let rounds_f = 8;
        let rounds_p = 56;
        let mut round_constants: Vec<[Fr; T]> = RC3
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

        let rounds_f_beginning = rounds_f / 2;
        let p_end = rounds_f_beginning + rounds_p;
        let internal_round_constants = round_constants
            .drain(rounds_f_beginning..p_end)
            .map(|vec| vec[0])
            .collect::<Vec<_>>();
        let external_round_constants = round_constants;
        Poseidon2Params::new(
            rounds_f,
            rounds_p,
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
