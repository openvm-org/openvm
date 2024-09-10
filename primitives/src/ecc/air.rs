use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use super::{columns::EcAddCols, EccAir};
use crate::{
    bigint::{DefaultLimbConfig, OverflowInt},
    sub_chip::AirConfig,
};

impl<F: Field> BaseAir<F> for EccAir {
    fn width(&self) -> usize {
        // x, y, lambda, and the quotients are size of num_limbs.
        // carries are 2 * carries - 1.
        self.num_limbs * 16 - 3
    }
}

impl AirConfig for EccAir {
    type Cols<T> = EcAddCols<T, DefaultLimbConfig>;
}

impl<AB: InteractionBuilder> Air<AB> for EccAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local = EcAddCols::<AB::Var, DefaultLimbConfig>::from_slice(&local, self.num_limbs);

        let EcAddCols { io, aux } = local;

        // λ = (y2 - y1) / (x2 - x1)
        let lambda =
            OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Var>(aux.lambda, self.limb_bits);
        let x1: OverflowInt<AB::Expr> = io.p1.x.into();
        let x2: OverflowInt<AB::Expr> = io.p2.x.into();
        let y1: OverflowInt<AB::Expr> = io.p1.y.into();
        let y2: OverflowInt<AB::Expr> = io.p2.y.into();
        let expr = lambda.clone() * (x2.clone() - x1.clone()) - y2 + y1.clone();
        self.check_carry
            .constrain_carry_mod_to_zero(builder, expr, aux.lambda_check);

        // x3 = λ * λ - x1 - x2
        let x3: OverflowInt<AB::Expr> = io.p3.x.into();
        let expr = lambda.clone() * lambda.clone() - x1.clone() - x2.clone() - x3.clone();
        self.check_carry
            .constrain_carry_mod_to_zero(builder, expr, aux.x3_check);

        // t = y1 - λ * x1
        // y3 = -(λ * x3 + t) = -λ * x3 - y1 + λ * x1
        let y3: OverflowInt<AB::Expr> = io.p3.y.into();
        let expr = y3 + lambda.clone() * x3 + y1 - lambda * x1;
        self.check_carry
            .constrain_carry_mod_to_zero(builder, expr, aux.y3_check);
    }
}
