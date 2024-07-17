use std::array::from_fn;

use p3_air::BaseAir;
use p3_field::{Field, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;

use afs_chips::sub_chip::LocalTraceInstructions;
use afs_test_utils::interaction::dummy_interaction_air::DummyInteractionAir;
use poseidon2_air::poseidon2::{Poseidon2Air, Poseidon2Config};

use crate::memory::expand::POSEIDON2_DIRECT_REQUEST_BUS;
use crate::memory::tree::HashProvider;

pub fn test_hash_sum<const CHUNK: usize, F: Field>(
    left: [F; CHUNK],
    right: [F; CHUNK],
) -> [F; CHUNK] {
    from_fn(|i| left[i] + right[i])
}

pub fn test_hash_poseidon2<F: PrimeField32>(left: [F; 8], right: [F; 8]) -> [F; 8] {
    let air =
        Poseidon2Air::<16, F>::from_config(Poseidon2Config::<16, F>::new_p3_baby_bear_16(), 0);
    let input_state = [left, right].concat().try_into().unwrap();
    let internal = air.generate_trace_row(input_state);
    let output = internal.io.output.to_vec();
    output[0..8].try_into().unwrap()
}

pub struct HashTestChip<const CHUNK: usize, F> {
    requests: Vec<[[F; CHUNK]; 3]>,
}

impl<const CHUNK: usize, F: Field> HashTestChip<CHUNK, F> {
    pub fn new() -> Self {
        Self { requests: vec![] }
    }

    pub fn air(&self) -> DummyInteractionAir {
        DummyInteractionAir::new(3 * CHUNK, false, POSEIDON2_DIRECT_REQUEST_BUS)
    }

    pub fn trace(&self) -> RowMajorMatrix<F> {
        let mut rows = vec![];
        for request in self.requests.iter() {
            rows.push(F::one());
            rows.extend(request.iter().flatten());
        }
        let width = BaseAir::<F>::width(&self.air());
        while !(rows.len() / width).is_power_of_two() {
            rows.push(F::zero());
        }
        RowMajorMatrix::new(rows, width)
    }
}

impl<const CHUNK: usize, F: Field> HashProvider<CHUNK, F> for HashTestChip<CHUNK, F> {
    fn hash(&mut self, left: [F; CHUNK], right: [F; CHUNK]) -> [F; CHUNK] {
        let result = test_hash_sum(left, right);
        self.requests.push([left, right, result]);
        result
    }
}
