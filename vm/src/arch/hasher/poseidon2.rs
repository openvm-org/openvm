use std::array;

use ax_poseidon2_air::poseidon2::Poseidon2Air;
use p3_field::{PrimeField, PrimeField32};

use crate::{
    arch::{
        hasher::Hasher, vm_poseidon2_config, DEFAULT_POSEIDON2_MAX_CONSTRAINT_DEGREE,
        POSEIDON2_WIDTH,
    },
    system::memory::CHUNK,
};

pub fn vm_poseidon2_hasher<F: PrimeField32>() -> Poseidon2Hasher<{ POSEIDON2_WIDTH }, F> {
    Poseidon2Hasher {
        poseidon2_air: Poseidon2Air::<POSEIDON2_WIDTH, F>::from_config(
            vm_poseidon2_config(),
            // `max_constraint_degree` and `bus_index` could be any value.
            DEFAULT_POSEIDON2_MAX_CONSTRAINT_DEGREE,
            0,
        ),
    }
}

pub struct Poseidon2Hasher<const WIDTH: usize, F: Clone> {
    poseidon2_air: Poseidon2Air<WIDTH, F>,
}

impl<F: PrimeField> Hasher<{ CHUNK }, F> for Poseidon2Hasher<{ POSEIDON2_WIDTH }, F> {
    fn compress(&self, lhs: &[F; CHUNK], rhs: &[F; CHUNK]) -> [F; CHUNK] {
        let mut input_state = [F::zero(); POSEIDON2_WIDTH];
        input_state[..CHUNK].copy_from_slice(lhs);
        input_state[CHUNK..].copy_from_slice(rhs);
        let inner_cols = self.poseidon2_air.generate_trace_row(input_state);
        array::from_fn(|i| inner_cols.io.output[i])
    }
}
