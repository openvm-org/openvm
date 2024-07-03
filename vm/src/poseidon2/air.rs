use std::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::PrimeField32;
use p3_matrix::Matrix;

use super::{columns::Poseidon2ChipCols, Poseidon2Chip};
use afs_chips::sub_chip::AirConfig;

impl<const WIDTH: usize, F: PrimeField32> AirConfig for Poseidon2Chip<WIDTH, F> {
    type Cols<T> = Poseidon2ChipCols<WIDTH, T>;
}

impl<const WIDTH: usize, F: PrimeField32> BaseAir<F> for Poseidon2Chip<WIDTH, F> {
    fn width(&self) -> usize {
        Poseidon2ChipCols::<WIDTH, F>::get_width(self)
    }
}

impl<AB: AirBuilder, const WIDTH: usize, F: PrimeField32> Air<AB> for Poseidon2Chip<WIDTH, F>
where
    Poseidon2Chip<WIDTH, F>: BaseAir<<AB as AirBuilder>::F>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let cols: &[<AB>::Var] = (*local).borrow();
        let cols = Poseidon2ChipCols::<WIDTH, AB::Var>::from_slice(cols, self);
    }
}
