use std::{ops::Neg, sync::Arc};

use num_bigint_dig::{BigInt, BigUint, Sign};
use num_traits::FromPrimitive;
use p3_field::PrimeField64;

use super::{
    columns::{EcAddAuxCols, EcAddCols, EcAddIoCols},
    EcPoint, EccAir,
};
use crate::{
    bigint::{
        check_carry_mod_to_zero::CheckCarryModToZeroCols, CanonicalUint, DefaultLimbConfig,
        OverflowInt,
    },
    range_gate::RangeCheckerGateChip,
    sub_chip::LocalTraceInstructions,
};

impl<F: PrimeField64> LocalTraceInstructions<F> for EccAir {
    type LocalInput = (
        (BigUint, BigUint),
        (BigUint, BigUint),
        Arc<RangeCheckerGateChip>,
    );

    fn generate_trace_row(&self, input: Self::LocalInput) -> Self::Cols<F> {
        // Assumes coordinates are within [0, p).
        let ((x1, y1), (x2, y2), range_checker) = input;
        assert_ne!(x1, x2);
        assert!(x1 < self.prime);
        assert!(x2 < self.prime);
        assert!(y1 < self.prime);
        assert!(y2 < self.prime);

        // helper functions ===
        let vec_isize_to_f = |x: Vec<isize>| {
            x.iter()
                .map(|x| {
                    F::from_canonical_usize(x.unsigned_abs())
                        * if x >= &0 { F::one() } else { F::neg_one() }
                })
                .collect()
        };
        let to_canonical = |x: &BigUint| {
            CanonicalUint::<isize, DefaultLimbConfig>::from_big_uint(x, Some(self.num_limbs))
        };
        let to_canonical_f = |x: &BigUint| {
            let limbs = vec_isize_to_f(to_canonical(x).limbs);
            CanonicalUint::<F, DefaultLimbConfig>::from_vec(limbs)
        };
        let bigint_to_limbs = |x: &BigInt| {
            let x_sign = x.sign();
            let x_abs_limbs = if x_sign == Sign::Plus {
                to_canonical(&x.to_biguint().unwrap()).limbs
            } else {
                to_canonical(&x.neg().to_biguint().unwrap()).limbs
            };
            if x_sign == Sign::Plus {
                x_abs_limbs
            } else {
                x_abs_limbs.iter().map(|x| -x).collect()
            }
        };
        let range_check = |bits: usize, value: usize| {
            let value = value as u32;
            if bits == self.decomp {
                range_checker.add_count(value);
            } else {
                range_checker.add_count(value);
                range_checker.add_count(value + (1 << self.decomp) - (1 << bits));
            }
        };
        // helper functions ===

        // Compute lambda and its carries: λ = (y2 - y1) / (x2 - x1)
        let dx = (self.prime.clone() + x2.clone() - x1.clone()) % self.prime.clone();
        let dy = (self.prime.clone() + y2.clone() - y1.clone()) % self.prime.clone();
        let exp = self.prime.clone() - BigUint::from_u8(2).unwrap();
        let dx_inv = dx.modpow(&exp, &self.prime);
        let lambda = dy.clone() * dx_inv % self.prime.clone();
        let lambda_canonical = to_canonical(&lambda);
        let dx_canonical = to_canonical(&dx);
        let dy_canonical = to_canonical(&dy);
        let expr: OverflowInt<isize> = OverflowInt::<isize>::from(lambda_canonical.clone())
            * dx_canonical.into()
            - dy_canonical.into();
        // TODO: Is the num limbs correct?
        let lambda_carries = expr.calculate_carries(self.limb_bits);
        // What's the bit range for carries?

        // dy is within [0, p), so lambda * dx - dy is non-negative.
        // (It's a multiple of p, and it's greater than -p).
        let lambda_q: BigUint = (lambda.clone() * dx - dy) / self.prime.clone();
        // TODO: Is the num limbs correct?
        let lambda_q = to_canonical(&lambda_q);
        for &q in lambda_q.limbs.iter() {
            range_check(self.limb_bits, q as usize);
        }

        // Compute x3 and its carries: x3 = λ * λ - x1 - x2
        // Adding prime twice to guarantee non-negative.
        let x3 = (lambda.clone() * lambda.clone() + self.prime.clone() + self.prime.clone()
            - x1.clone()
            - x2.clone())
            % self.prime.clone();
        let x3_canonical = to_canonical(&x3);
        let x1_canonical = to_canonical(&x1);
        let x2_canonical = to_canonical(&x2);
        let expr: OverflowInt<isize> = OverflowInt::<isize>::from(lambda_canonical.clone())
            * lambda_canonical.clone().into()
            - x3_canonical.clone().into()
            - x1_canonical.clone().into()
            - x2_canonical.clone().into();
        // TODO: check carries num limbs.
        let x3_carries = expr.calculate_carries(self.limb_bits);
        // We don't know if it's positive or negative.
        let x3_q_signed = (BigInt::from_biguint(Sign::Plus, lambda.clone() * lambda.clone())
            - BigInt::from_biguint(Sign::Plus, x1.clone())
            - BigInt::from_biguint(Sign::Plus, x2.clone())
            - BigInt::from_biguint(Sign::Plus, x3.clone()))
            / BigInt::from_biguint(Sign::Plus, self.prime.clone());
        let x3_q_limbs = bigint_to_limbs(&x3_q_signed);

        // Compute y3 and its carries: y3 = -λ * x3 - y1 + λ * x1.
        // Adding prime to guarantee non-negative.
        let y3 = ((self.prime.clone() + x1.clone() - x3.clone()) * lambda.clone()
            + self.prime.clone()
            - y1.clone())
            % self.prime.clone();
        let y3_canonical = to_canonical(&y3);
        let y1_canonical = to_canonical(&y1);
        let expr: OverflowInt<isize> = OverflowInt::<isize>::from(y3_canonical)
            + OverflowInt::<isize>::from(lambda_canonical.clone()) * x3_canonical.into()
            + y1_canonical.into()
            - OverflowInt::<isize>::from(lambda_canonical.clone()) * x1_canonical.into();
        let y3_carries = expr.calculate_carries(self.limb_bits);
        let y3_q_signed = (BigInt::from_biguint(Sign::Plus, y3.clone())
            + BigInt::from_biguint(Sign::Plus, lambda.clone() * x3.clone())
            + BigInt::from_biguint(Sign::Plus, y1.clone())
            - BigInt::from_biguint(Sign::Plus, lambda * x1.clone()))
            / BigInt::from_biguint(Sign::Plus, self.prime.clone());
        let y3_q_limbs = bigint_to_limbs(&y3_q_signed);

        let io = EcAddIoCols {
            p1: EcPoint {
                x: to_canonical_f(&x1),
                y: to_canonical_f(&y1),
            },
            p2: EcPoint {
                x: to_canonical_f(&x2),
                y: to_canonical_f(&y2),
            },
            p3: EcPoint {
                x: to_canonical_f(&x3),
                y: to_canonical_f(&y3),
            },
        };

        let aux = EcAddAuxCols {
            lambda: vec_isize_to_f(lambda_canonical.limbs),
            lambda_check: CheckCarryModToZeroCols {
                carries: vec_isize_to_f(lambda_carries),
                quotient: vec_isize_to_f(lambda_q.limbs),
            },
            x3_check: CheckCarryModToZeroCols {
                carries: vec_isize_to_f(x3_carries),
                quotient: vec_isize_to_f(x3_q_limbs),
            },
            y3_check: CheckCarryModToZeroCols {
                carries: vec_isize_to_f(y3_carries),
                quotient: vec_isize_to_f(y3_q_limbs),
            },
        };

        EcAddCols { io, aux }
    }
}
