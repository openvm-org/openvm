use std::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use afs_chips::is_zero::columns::IsZeroIOCols;
use afs_chips::is_zero::IsZeroAir;
use afs_chips::sub_chip::AirConfig;
use afs_chips::sub_chip::SubAir;
use poseidon2_air::poseidon2::columns::Poseidon2Cols;

use super::{columns::Poseidon2ChipCols, Poseidon2Chip};

impl<const WIDTH: usize, F: Clone> AirConfig for Poseidon2Chip<WIDTH, F> {
    type Cols<T> = Poseidon2ChipCols<WIDTH, T>;
}

impl<const WIDTH: usize, F: Field> BaseAir<F> for Poseidon2Chip<WIDTH, F> {
    fn width(&self) -> usize {
        Poseidon2ChipCols::<WIDTH, F>::get_width(self)
    }
}

impl<AB: AirBuilder, const WIDTH: usize> Air<AB> for Poseidon2Chip<WIDTH, AB::F> {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &[<AB>::Var] = (*local).borrow();

        let index_map = Poseidon2Cols::index_map(&self.air);
        let cols = Poseidon2ChipCols::<WIDTH, AB::Var>::from_slice(local, &index_map);
        SubAir::<AB>::eval(
            &self.air,
            builder,
            cols.aux.internal.io,
            cols.aux.internal.aux,
        );
        // boolean constraints for alloc/cmp markers
        builder.assert_bool(cols.io.is_alloc);
        builder.assert_bool(cols.io.cmp);
        // can only be comparing if row is allocated
        builder.assert_eq(cols.io.is_alloc * cols.io.cmp, cols.io.cmp);
        // immediates
        for (i, operand) in [cols.io.a, cols.io.b, cols.io.c].into_iter().enumerate() {
            builder
                .when(cols.aux.d_is_zero)
                .assert_eq(cols.aux.addresses[i], operand);
        }
        // d is zero SubAir
        SubAir::eval(
            &IsZeroAir {},
            builder,
            IsZeroIOCols {
                x: cols.io.d,
                is_zero: cols.aux.d_is_zero,
            },
            cols.aux.is_zero_inv,
        );
    }
}
