use std::array::from_fn;

use afs_test_utils::interaction::dummy_interaction_air::DummyInteractionAir;
use p3_air::BaseAir;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

use crate::memory::{expand::POSEIDON2_DIRECT_REQUEST_BUS, tree::Hasher};

pub fn test_hash_sum<const CHUNK: usize, F: Field>(
    left: [F; CHUNK],
    right: [F; CHUNK],
) -> [F; CHUNK] {
    from_fn(|i| left[i] + right[i])
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

impl<const CHUNK: usize, F: Field> Hasher<CHUNK, F> for HashTestChip<CHUNK, F> {
    fn hash(&mut self, left: [F; CHUNK], right: [F; CHUNK]) -> [F; CHUNK] {
        let result = test_hash_sum(left, right);
        self.requests.push([left, right, result]);
        result
    }
}
