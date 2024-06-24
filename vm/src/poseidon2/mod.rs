pub mod air;
pub mod columns;
pub mod trace;

use p3_baby_bear::BabyBear;
use p3_field::PrimeField;

use self::columns::Poseidon2Cols;

// use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};

pub struct Poseidon2Air<const WIDTH: usize, T: PrimeField> {
    pub rounds_f: usize,
    pub external_constants: Vec<[T; WIDTH]>,
    pub rounds_p: usize,
    pub internal_constants: Vec<T>,
}

impl<const WIDTH: usize, T: PrimeField> Poseidon2Air<WIDTH, T> {
    pub fn new(external_constants: Vec<[T; WIDTH]>, internal_constants: Vec<T>) -> Self {
        Self {
            rounds_f: external_constants.len(),
            external_constants,
            rounds_p: internal_constants.len(),
            internal_constants,
        }
    }
    pub fn get_width(&self) -> usize {
        Poseidon2Cols::<WIDTH, T>::get_width(self)
    }
}
