use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use crate::sub_chip::AirConfig;

use super::{
    columns::{FlatHashCols, FlatHashInternalCols},
    FlatHashAir,
};

impl<F: Field> BaseAir<F> for FlatHashAir {
    fn width(&self) -> usize {
        self.get_width()
    }
}

impl AirConfig for FlatHashAir {
    type Cols<T> = FlatHashCols<T>;
}

impl<AB: AirBuilderWithPublicValues> Air<AB> for FlatHashAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let pis = builder.public_values();

        let digest_start = self.page_width / self.hash_rate * self.hash_width;

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local_flat_hash_cols: &FlatHashCols<_> =
            &FlatHashCols::from_slice(local.as_ref(), self);
        let next_flat_hash_cols: &FlatHashCols<_> = &FlatHashCols::from_slice(next.as_ref(), self);

        let next_is_alloc = next_flat_hash_cols.io.is_alloc;

        let FlatHashInternalCols {
            hashes: local_hashes,
        } = local_flat_hash_cols.aux.clone();

        let FlatHashInternalCols {
            hashes: next_hashes,
        } = next_flat_hash_cols.aux.clone();

        // First loop for immutable borrow
        let digest_assertions: Vec<_> = (0..self.digest_width)
            .map(|i| (local_hashes[digest_start + i], pis[i]))
            .collect();

        // Second loop for mutable borrow
        for (local_hash, pi) in digest_assertions {
            builder.when_last_row().assert_eq(local_hash, pi);
        }

        for local_hash in local_hashes.iter().take(self.hash_width) {
            builder.when_first_row().assert_zero(*local_hash);
        }

        let mut transition = builder.when_transition();
        let last_row_index = (self.page_width / self.hash_rate) * self.hash_width;

        for i in 0..self.hash_width {
            transition
                .assert_zero(next_is_alloc * (local_hashes[i + last_row_index] - next_hashes[i]));
        }
        for i in 0..self.digest_width {
            transition.assert_zero(
                (AB::Expr::one() - next_is_alloc)
                    * (next_hashes[digest_start + i] - local_hashes[digest_start + i]),
            );
        }
    }
}
