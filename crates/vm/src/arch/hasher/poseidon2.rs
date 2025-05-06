use std::{
    array::{self, from_fn},
    marker::PhantomData,
};

use openvm_poseidon2_air::p3_symmetric::Permutation;
use openvm_stark_backend::p3_field::{FieldAlgebra, PrimeField32};
use openvm_stark_sdk::p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};

use crate::{
    arch::{hasher::Hasher, vm_poseidon2_config, POSEIDON2_WIDTH},
    system::memory::CHUNK,
};

pub fn vm_poseidon2_hasher<F: PrimeField32>() -> Poseidon2Hasher<F> {
    assert_eq!(F::ORDER_U32, KoalaBear::ORDER_U32, "F must be KoalaBear");
    let config = vm_poseidon2_config::<KoalaBear>();
    let (external_constants, internal_constants) =
        config.constants.to_external_internal_constants();
    Poseidon2Hasher {
        poseidon2: Poseidon2KoalaBear::new(external_constants, internal_constants),
        _marker: PhantomData,
    }
}

/// `F` must be KoalaBear. Don't use this for anything performance sensitive.
pub struct Poseidon2Hasher<F: Clone> {
    poseidon2: Poseidon2KoalaBear<POSEIDON2_WIDTH>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField32> Hasher<{ CHUNK }, F> for Poseidon2Hasher<F> {
    fn compress(&self, lhs: &[F; CHUNK], rhs: &[F; CHUNK]) -> [F; CHUNK] {
        let mut state = from_fn(|i| {
            if i < CHUNK {
                KoalaBear::from_canonical_u32(lhs[i].as_canonical_u32())
            } else {
                KoalaBear::from_canonical_u32(rhs[i - CHUNK].as_canonical_u32())
            }
        });
        self.poseidon2.permute_mut(&mut state);
        array::from_fn(|i| F::from_canonical_u32(state[i].as_canonical_u32()))
    }
}
