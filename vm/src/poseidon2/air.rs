use p3_air::Air;
use p3_baby_bear::BabyBear;
use p3_poseidon2::Poseidon2Air;

pub struct Poseidon2Air<const WIDTH: usize> {
    pub air: Air<BabyBear, WIDTH>,
}
