use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::{columns::EcAddCols, EccAir};
use crate::{
    bigint::{DefaultLimbConfig, OverflowInt},
    sub_chip::AirConfig,
};

impl<F: Field> BaseAir<F> for EccAir {
    fn width(&self) -> usize {
        // x, y, lambda, and the quotients are size of num_limbs.
        // signs size 1.
        // carries are 2 * carries - 1.
        self.num_limbs * 16
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
        // lambda_expr_sign is 1 for negative, 0 for non-negative
        // 1 - 2x will be     -1               1
        builder.assert_bool(aux.lambda_expr_sign);
        let sign: AB::Expr =
            AB::Expr::from_canonical_u8(1) - AB::Expr::from_canonical_u8(2) * aux.lambda_expr_sign;
        let sign = OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Expr>(vec![sign], 1);
        let expr: OverflowInt<AB::Expr> =
            (lambda.clone() * (x2.clone() - x1.clone()) - y2 + y1.clone()) * sign;
        self.check_carry
            .constrain_carry_mod_to_zero(builder, expr, aux.lambda_check);

        // x3 = λ * λ - x1 - x2
        let x3: OverflowInt<AB::Expr> = io.p3.x.into();
        builder.assert_bool(aux.x3_expr_sign);
        let sign: AB::Expr =
            AB::Expr::from_canonical_u8(1) - AB::Expr::from_canonical_u8(2) * aux.x3_expr_sign;
        let sign = OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Expr>(vec![sign], 1);
        let expr = (lambda.clone() * lambda.clone() - x1.clone() - x2.clone() - x3.clone()) * sign;
        self.check_carry
            .constrain_carry_mod_to_zero(builder, expr, aux.x3_check);

        // t = y1 - λ * x1
        // y3 = -(λ * x3 + t) = -λ * x3 - y1 + λ * x1
        let y3: OverflowInt<AB::Expr> = io.p3.y.into();
        builder.assert_bool(aux.y3_expr_sign);
        let sign: AB::Expr =
            AB::Expr::from_canonical_u8(1) - AB::Expr::from_canonical_u8(2) * aux.y3_expr_sign;
        let sign = OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Expr>(vec![sign], 1);
        let expr = (y3 + lambda.clone() * x3 + y1 - lambda * x1) * sign;
        self.check_carry
            .constrain_carry_mod_to_zero(builder, expr, aux.y3_check);
    }
}
