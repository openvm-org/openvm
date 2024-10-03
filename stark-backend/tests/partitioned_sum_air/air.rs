//! AIR with partitioned main trace
//! | x | y_0 | ... | y_w |
//!
//! Constrains x == a_0 + ... + a_w

use afs_stark_backend::{
    air_builders::PartitionedAirBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, BaseAir};
use p3_field::AbstractField;
use p3_matrix::Matrix;

/// Inner value is width of y-submatrix
pub struct SumAir(pub usize);

impl<F> BaseAirWithPublicValues<F> for SumAir {}
impl<F> PartitionedBaseAir<F> for SumAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![self.0]
    }
    fn common_main_width(&self) -> usize {
        1
    }
}
impl<F> BaseAir<F> for SumAir {
    fn width(&self) -> usize {
        self.0 + 1
    }
}

impl<AB: PartitionedAirBuilder> Air<AB> for SumAir {
    fn eval(&self, builder: &mut AB) {
        assert_eq!(builder.cached_mains().len(), 1);

        let x = builder.common_main().row_slice(0)[0];
        let ys = builder.cached_mains()[0].row_slice(0);

        let mut y_sum = AB::Expr::zero();
        for &y in &*ys {
            y_sum = y_sum + y;
        }
        drop(ys);

        builder.assert_eq(x, y_sum);
    }
}
