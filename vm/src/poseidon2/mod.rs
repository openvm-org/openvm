pub mod air;
pub mod columns;
pub mod trace;

use p3_baby_bear::BabyBear;

// use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};

pub struct Poseidon2Air {
    pub rounds_f: usize,
    pub external_constants: Vec<[BabyBear; 16]>,
    pub rounds_p: usize,
    pub internal_constants: Vec<BabyBear>,
}

impl Poseidon2Air {
    pub fn new(external_constants: Vec<[BabyBear; 16]>, internal_constants: Vec<BabyBear>) -> Self {
        Self {
            rounds_f: external_constants.len(),
            external_constants,
            rounds_p: internal_constants.len(),
            internal_constants,
        }
    }
}
