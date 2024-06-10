use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use crate::sub_chip::AirConfig;

use super::{
    columns::{FlatHashCols, FlatHashInternalCols},
    FlatHashChip,
};

impl<F: Field> BaseAir<F> for FlatHashChip {
    fn width(&self) -> usize {
        self.page_width + (self.page_width / self.hash_rate + 1) * self.hash_width
    }
}

impl AirConfig for FlatHashChip {
    type Cols<T> = FlatHashCols<T>;
}

impl<AB: AirBuilder> Air<AB> for FlatHashChip {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let local_flat_hash_cols: &FlatHashCols<_> = &FlatHashCols::from_slice(
            local.as_ref(),
            self.page_width,
            self.page_height,
            self.hash_width,
            self.hash_rate,
            self.digest_width,
        );

        let next_flat_hash_cols: &FlatHashCols<_> = &FlatHashCols::from_slice(
            next.as_ref(),
            self.page_width,
            self.page_height,
            self.hash_width,
            self.hash_rate,
            self.digest_width,
        );

        let FlatHashInternalCols {
            hashes: local_hashes,
        } = local_flat_hash_cols.aux.clone();

        let FlatHashInternalCols {
            hashes: next_hashes,
        } = next_flat_hash_cols.aux.clone();

        let mut first_row = builder.when_first_row();

        for local_hash in local_hashes.iter().take(self.hash_width) {
            first_row.assert_zero(*local_hash);
        }

        let mut transition = builder.when_transition();
        let last_row_index = (self.page_width / self.hash_rate) * self.hash_width;

        for i in 0..self.hash_width {
            transition.assert_eq(local_hashes[i + last_row_index], next_hashes[i]);
        }
    }
}
