use super::Poseidon2Air;
use p3_air::Air;
use p3_air::AirBuilder;
use p3_air::BaseAir;
use p3_baby_bear::BabyBear;
use p3_field::Field;

impl<const WIDTH: usize, F: Field> BaseAir<F> for Poseidon2Air<WIDTH> {
    fn width(&self) -> usize {
        self.get_width()
    }
}

impl<AB: AirBuilder, const WIDTH: usize> Air<AB> for Poseidon2Air<WIDTH> {
    fn eval(&self, builder: &mut AB) {
        todo!()
    }
}
