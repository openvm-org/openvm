use p3_field::{AbstractField, PrimeField64};

use super::{Array, DslIr};
use crate::ir::{modular_arithmetic::BigUintVar, Builder, Config, Var};

impl<C: Config> Builder<C>
where
    C::N: PrimeField64,
{
    /// Computes `p + q`, handling cases where `p` or `q` are identity.
    ///
    /// A point is stored as a tuple of affine coordinates, contiguously in memory as 64 bytes.
    /// Identity point is represented as (0, 0).
    pub fn secp256k1_add(
        &mut self,
        point_1: Array<C, Var<C::N>>,
        point_2: Array<C, Var<C::N>>,
    ) -> Array<C, Var<C::N>> {
        // number of limbs to represent one coordinate
        let num_limbs = ((256 + self.bigint_repr_size - 1) / self.bigint_repr_size) as usize;
        // Assuming point_1.len() = 2 * num_limbs
        let x1 = point_1.slice(self, 0, num_limbs);
        let y1 = point_1.slice(self, num_limbs, 2 * num_limbs);

        let res = self.uninit();
        let x1_zero = self.secp256k1_coord_is_zero(&x1);
        let y1_zero = self.secp256k1_coord_is_zero(&y1);

        // if point_1 is identity
        self.if_eq(x1_zero * y1_zero, C::N::one()).then_or_else(
            |builder| {
                builder.assign(&res, point_2.clone());
            },
            |builder| {
                let x2 = point_2.slice(builder, 0, num_limbs);
                let y2 = point_2.slice(builder, num_limbs, 2 * num_limbs);
                let x2_zero = builder.secp256k1_coord_is_zero(&x2);
                let y2_zero = builder.secp256k1_coord_is_zero(&y2);
                // else if point_2 is identity
                builder.if_eq(x2_zero * y2_zero, C::N::one()).then_or_else(
                    |builder| {
                        builder.assign(&res, point_1.clone());
                    },
                    |builder| {
                        let xs_equal = builder.secp256k1_coord_eq(&x1, &x2);
                        builder.if_eq(xs_equal, C::N::one()).then_or_else(
                            |builder| {
                                // if x1 == x2
                                let ys_equal = builder.secp256k1_coord_eq(&y1, &y2);
                                builder.if_eq(ys_equal, C::N::one()).then_or_else(
                                    |builder| {
                                        // if y1 == y2 => point_1 == point_2, do double
                                        let res_double = builder.secp256k1_double(point_1.clone());
                                        builder.assign(&res, res_double);
                                    },
                                    |builder| {
                                        // else y1 != y2 => x1 = x2, y1 = - y2 so point_1 + point_2 = identity
                                        let identity = builder.array(2 * num_limbs);
                                        for i in 0..2 * num_limbs {
                                            builder.set(&identity, i, C::N::zero());
                                        }
                                        builder.assign(&res, identity)
                                    },
                                )
                            },
                            |builder| {
                                // if x1 != x2
                                let res_ne =
                                    builder.secp256k1_add_unequal(point_1.clone(), point_2.clone());
                                builder.assign(&res, res_ne);
                            },
                        )
                    },
                )
            },
        );
        res
    }

    /// Assumes that `point_1 != +- point_2` which is equivalent to `point_1.x != point_2.x`.
    /// Does not handle identity points.
    ///
    /// A point is stored as a tuple of affine coordinates, contiguously in memory as 64 bytes.
    pub fn secp256k1_add_unequal(
        &mut self,
        point_1: Array<C, Var<C::N>>,
        point_2: Array<C, Var<C::N>>,
    ) -> Array<C, Var<C::N>> {
        // TODO: enforce this is constant length
        let dst = self.array(point_1.len());
        self.push(DslIr::Secp256k1AddUnequal(dst.clone(), point_1, point_2));
        dst
    }

    /// Does not handle identity points.
    ///
    /// A point is stored as a tuple of affine coordinates, contiguously in memory as 64 bytes.
    pub fn secp256k1_double(&mut self, point: Array<C, Var<C::N>>) -> Array<C, Var<C::N>> {
        let dst = self.array(point.len());
        self.push(DslIr::Secp256k1Double(dst.clone(), point));
        dst
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
