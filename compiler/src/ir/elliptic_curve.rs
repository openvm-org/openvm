use num_bigint_dig::BigUint;
use num_traits::{FromPrimitive, Zero};
use p3_field::{AbstractField, PrimeField64};

use crate::ir::{Builder, Config, modular_arithmetic::BigIntVar};

const EC_CONSTANT: usize = 4;

impl<C: Config> Builder<C>
where
    C::N: PrimeField64,
{
    pub fn ec_add(
        &mut self,
        left: &(BigIntVar<C>, BigIntVar<C>),
        right: &(BigIntVar<C>, BigIntVar<C>),
    ) -> (BigIntVar<C>, BigIntVar<C>) {
        let (x1, y1) = left;
        let (x2, y2) = right;

        let xs_equal = self.bigint_eq(&x1, &x2);
        let ys_equal = self.bigint_eq(&y1, &y2);
        let y_sum = self.mod_add(&y1, &y2);
        let ys_opposite = self.bigint_is_zero(&y_sum);
        let result_x = self.uninit();
        let result_y = self.uninit();

        self.if_eq(xs_equal * ys_opposite, C::N::one())
            .then_or_else(
                |builder| {
                    let zero = builder.eval_bigint(BigUint::zero());
                    builder.assign(&result_x, zero.clone());
                    builder.assign(&result_y, zero);
                },
                |builder| {
                    let lambda = builder.uninit();
                    builder
                        .if_eq(xs_equal * ys_equal, C::N::one())
                        .then_or_else(
                            |builder| {
                                let two = builder.eval_bigint(BigUint::from_u8(2).unwrap());
                                let three = builder.eval_bigint(BigUint::from_u8(3).unwrap());
                                let two_y = builder.mod_mul(&two, &y1);
                                let x_squared = builder.mod_mul(&x1, &x1);
                                let three_x_squared = builder.mod_mul(&three, &x_squared);
                                let lambda_value = builder.mod_div(&three_x_squared, &two_y);
                                builder.assign(&lambda, lambda_value);
                            },
                            |builder| {
                                let dy = builder.mod_sub(&y2, &y1);
                                let dx = builder.mod_sub(&x2, &x1);
                                let lambda_value = builder.mod_div(&dy, &dx);
                                builder.assign(&lambda, lambda_value);
                            },
                        );
                    let lambda_squared = builder.mod_mul(&lambda, &lambda);
                    let x_sum = builder.mod_add(&x1, &x2);
                    let x3 = builder.mod_sub(&lambda_squared, &x_sum);
                    let x1_minus_x3 = builder.mod_sub(&x1, &x3);
                    let code_is_self_documenting = builder.mod_mul(&lambda, &x1_minus_x3);
                    let y3 = builder.mod_sub(&code_is_self_documenting, &y1);
                    builder.assign(&result_x, x3);
                    builder.assign(&result_y, y3);
                },
            );

        (result_x, result_y)
    }
}
