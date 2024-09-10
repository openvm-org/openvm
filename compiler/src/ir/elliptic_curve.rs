use num_bigint_dig::BigUint;
use num_traits::{FromPrimitive, Zero};
use p3_field::{AbstractField, PrimeField64};

use crate::ir::{modular_arithmetic::BigUintVar, Builder, Config, Var};

impl<C: Config> Builder<C>
where
    C::N: PrimeField64,
{
    pub fn ec_add(
        &mut self,
        point_1: &(BigUintVar<C>, BigUintVar<C>),
        point_2: &(BigUintVar<C>, BigUintVar<C>),
    ) -> (BigUintVar<C>, BigUintVar<C>) {
        let (x1, y1) = point_1;
        let (x2, y2) = point_2;

        let x1_zero = self.secp256k1_coord_is_zero(x1);
        let y1_zero = self.secp256k1_coord_is_zero(y1);
        let x2_zero = self.secp256k1_coord_is_zero(x2);
        let y2_zero = self.secp256k1_coord_is_zero(y2);
        let xs_equal = self.secp256k1_coord_eq(x1, x2);
        let ys_equal = self.secp256k1_coord_eq(y1, y2);
        let y_sum = self.secp256k1_coord_add(y1, y2);
        let ys_opposite = self.secp256k1_coord_is_zero(&y_sum);
        let result_x = self.uninit();
        let result_y = self.uninit();

        // if point_1 is identity
        self.if_eq(x1_zero * y1_zero, C::N::one()).then_or_else(
            |builder| {
                builder.assign(&result_x, x2.clone());
                builder.assign(&result_y, y2.clone());
            },
            |builder| {
                // else if point_2 is identity
                builder.if_eq(x2_zero * y2_zero, C::N::one()).then_or_else(
                    |builder| {
                        builder.assign(&result_x, x1.clone());
                        builder.assign(&result_y, y1.clone());
                    },
                    |builder| {
                        // else if point_1 = -point_2
                        builder
                            .if_eq(xs_equal * ys_opposite, C::N::one())
                            .then_or_else(
                                |builder| {
                                    let zero = builder.eval_biguint(BigUint::zero());
                                    builder.assign(&result_x, zero.clone());
                                    builder.assign(&result_y, zero);
                                },
                                |builder| {
                                    let lambda = builder.uninit();
                                    // else if point_1 = point_2
                                    builder
                                        .if_eq(xs_equal * ys_equal, C::N::one())
                                        .then_or_else(
                                            |builder| {
                                                let two = builder
                                                    .eval_biguint(BigUint::from_u8(2).unwrap());
                                                let three = builder
                                                    .eval_biguint(BigUint::from_u8(3).unwrap());
                                                let two_y = builder.secp256k1_coord_mul(&two, y1);
                                                let x_squared = builder.secp256k1_coord_mul(x1, x1);
                                                let three_x_squared =
                                                    builder.secp256k1_coord_mul(&three, &x_squared);
                                                let lambda_value = builder
                                                    .secp256k1_coord_div(&three_x_squared, &two_y);
                                                builder.assign(&lambda, lambda_value);
                                            },
                                            |builder| {
                                                // else (general case)
                                                let dy = builder.secp256k1_coord_sub(y2, y1);
                                                let dx = builder.secp256k1_coord_sub(x2, x1);
                                                let lambda_value =
                                                    builder.secp256k1_coord_div(&dy, &dx);
                                                builder.assign(&lambda, lambda_value);
                                            },
                                        );
                                    let lambda_squared =
                                        builder.secp256k1_coord_mul(&lambda, &lambda);
                                    let x_sum = builder.secp256k1_coord_add(x1, x2);
                                    let x3 = builder.secp256k1_coord_sub(&lambda_squared, &x_sum);
                                    let x1_minus_x3 = builder.secp256k1_coord_sub(x1, &x3);
                                    let lambda_times_x1_minus_x3 =
                                        builder.secp256k1_coord_mul(&lambda, &x1_minus_x3);
                                    let y3 =
                                        builder.secp256k1_coord_sub(&lambda_times_x1_minus_x3, y1);
                                    builder.assign(&result_x, x3);
                                    builder.assign(&result_y, y3);
                                },
                            );
                    },
                )
            },
        );

        (result_x, result_y)
    }

    /// Assert (x, y) is on the curve.
    pub fn ec_is_on_curve(&mut self, x: &BigUintVar<C>, y: &BigUintVar<C>) -> Var<C::N> {
        let x2 = self.secp256k1_coord_mul(x, x);
        let x3 = self.secp256k1_coord_mul(&x2, x);
        let c7 = self.eval_biguint(7u64.into());
        let x3_plus_7 = self.secp256k1_coord_add(&x3, &c7);
        let y2 = self.secp256k1_coord_mul(y, y);
        self.secp256k1_coord_eq(&y2, &x3_plus_7)
    }
}
