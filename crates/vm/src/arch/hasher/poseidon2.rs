use std::{
    array::{self, from_fn},
    marker::PhantomData,
};

use ax_poseidon2_air::{p3_poseidon2::ExternalLayerConstants, p3_symmetric::Permutation};
use ax_stark_sdk::p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use ax_stark_backend::p3_field::{AbstractField, PrimeField32};

use crate::{
    arch::{hasher::Hasher, vm_poseidon2_config, POSEIDON2_WIDTH},
    system::memory::CHUNK,
};

pub fn vm_poseidon2_hasher<F: PrimeField32>() -> Poseidon2Hasher<F> {
    assert_eq!(F::ORDER_U32, BabyBear::ORDER_U32, "F must be BabyBear");
    let config = vm_poseidon2_config::<BabyBear>();
    let external_constants = ExternalLayerConstants::new(
        config.external_constants[..config.rounds_f() / 2].to_vec(),
        config.external_constants[config.rounds_f() / 2..].to_vec(),
    );
    Poseidon2Hasher {
        poseidon2: Poseidon2BabyBear::new(external_constants, config.internal_constants),
        _marker: PhantomData,
    }
}

/// `F` must be BabyBear. Don't use this for anything performance sensitive.
pub struct Poseidon2Hasher<F: Clone> {
    poseidon2: Poseidon2BabyBear<POSEIDON2_WIDTH>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField32> Hasher<{ CHUNK }, F> for Poseidon2Hasher<F> {
    fn compress(&self, lhs: &[F; CHUNK], rhs: &[F; CHUNK]) -> [F; CHUNK] {
        let mut state = from_fn(|i| {
            if i < CHUNK {
                BabyBear::from_canonical_u32(lhs[i].as_canonical_u32())
            } else {
                BabyBear::from_canonical_u32(rhs[i - CHUNK].as_canonical_u32())
            }
        });
        self.poseidon2.permute_mut(&mut state);
        array::from_fn(|i| F::from_canonical_u32(state[i].as_canonical_u32()))
    }
}
