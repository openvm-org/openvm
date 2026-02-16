use std::{array::from_fn, sync::Mutex};

use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::Field,
    p3_matrix::dense::RowMajorMatrix,
    prover::{AirProvingContext, ColMajorMatrix, CpuBackend},
    test_utils::dummy_airs::interaction::dummy_interaction_air::DummyInteractionAir,
    StarkProtocolConfig,
};

use crate::arch::{
    hasher::{Hasher, HasherChip},
    testing::POSEIDON2_DIRECT_BUS,
};

pub fn test_hash_sum<const CHUNK: usize, F: Field>(
    left: [F; CHUNK],
    right: [F; CHUNK],
) -> [F; CHUNK] {
    from_fn(|i| left[i] + right[i])
}

pub struct HashTestChip<const CHUNK: usize, F> {
    requests: Mutex<Vec<[[F; CHUNK]; 3]>>,
}

impl<const CHUNK: usize, F: Field> HashTestChip<CHUNK, F> {
    pub fn new() -> Self {
        Self {
            requests: Mutex::new(vec![]),
        }
    }

    pub fn air(&self) -> DummyInteractionAir {
        DummyInteractionAir::new(3 * CHUNK, false, POSEIDON2_DIRECT_BUS)
    }

    pub fn trace(&self) -> RowMajorMatrix<F> {
        let mut rows = vec![];
        let requests = self.requests.lock().expect("mutex poisoned");
        for request in requests.iter() {
            rows.push(F::ONE);
            rows.extend(request.iter().flatten());
        }
        let width = BaseAir::<F>::width(&self.air());
        while !(rows.len() / width).is_power_of_two() {
            rows.push(F::ZERO);
        }
        RowMajorMatrix::new(rows, width)
    }
    pub fn generate_proving_ctx<SC>(&mut self) -> AirProvingContext<CpuBackend<SC>>
    where
        SC: StarkProtocolConfig<F = F>,
    {
        let trace = self.trace();
        AirProvingContext::simple_no_pis(ColMajorMatrix::from_row_major(&trace))
    }
}

impl<const CHUNK: usize, F: Field> Hasher<CHUNK, F> for HashTestChip<CHUNK, F> {
    fn compress(&self, left: &[F; CHUNK], right: &[F; CHUNK]) -> [F; CHUNK] {
        test_hash_sum(*left, *right)
    }
}

impl<const CHUNK: usize, F: Field> HasherChip<CHUNK, F> for HashTestChip<CHUNK, F> {
    fn compress_and_record(&self, left: &[F; CHUNK], right: &[F; CHUNK]) -> [F; CHUNK] {
        let result = test_hash_sum(*left, *right);
        let mut requests = self.requests.lock().expect("mutex poisoned");
        requests.push([*left, *right, result]);
        result
    }
}
