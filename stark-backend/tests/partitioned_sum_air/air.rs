//! AIR with partitioned main trace
//! | x | y_0 | ... | y_w |
//!
//! Constrains x == a_0 + ... + a_w

use afs_stark_backend::{air_builders::PartitionedAirBuilder, interaction::Chip};
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

/// Inner value is width of y-submatrix
pub struct SumAir(pub usize);

impl<F: Field> Chip<F> for SumAir {}

impl<F> BaseAir<F> for SumAir {
    fn width(&self) -> usize {
        self.0 + 1
    }
}

impl<AB: PartitionedAirBuilder> Air<AB> for SumAir {
    fn eval(&self, builder: &mut AB) {
        let partitioned_main = builder.partitioned_main();
        assert_eq!(partitioned_main.len(), 2);

        let x = partitioned_main[0].row_slice(0)[0];
        let ys = partitioned_main[1].row_slice(0);

        let mut y_sum = AB::Expr::zero();
        for &y in &*ys {
            y_sum = y_sum + y;
        }
        drop(ys);

        builder.assert_eq(x, y_sum);
    }
}
