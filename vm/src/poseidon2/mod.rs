pub mod air;
pub mod columns;
pub mod trace;

use p3_baby_bear::BabyBear;

use self::columns::Poseidon2Cols;

// use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};

pub struct Poseidon2Air<const WIDTH: usize> {
    pub rounds_f: usize,
    pub external_constants: Vec<Vec<BabyBear>>,
    pub rounds_p: usize,
    pub internal_constants: Vec<BabyBear>,
}

impl<const WIDTH: usize> Poseidon2Air<WIDTH> {
    pub fn new(external_constants: Vec<Vec<BabyBear>>, internal_constants: Vec<BabyBear>) -> Self {
        Self {
            rounds_f: external_constants.len(),
            external_constants,
            rounds_p: internal_constants.len(),
            internal_constants,
        }
    }
    pub fn get_width(&self) -> usize {
        Poseidon2Cols::<WIDTH>::get_width(self)
    }
}
