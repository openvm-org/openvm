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

        // ===== helper functions =====
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
        let to_overflow_int = |x: &BigUint| OverflowInt::<isize>::from(to_canonical(x));
        // Make the limbs for the absolute value of a bigint.
        let bigint_abs = |x: &BigInt| {
            if x.sign() == Sign::Minus {
                x.neg().to_biguint().unwrap()
            } else {
                x.to_biguint().unwrap()
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

        // ===== λ =====
        // Compute lambda: λ = (y2 - y1) / (x2 - x1).
        let dx = (self.prime.clone() + x2.clone() - x1.clone()) % self.prime.clone();
        let dy = (self.prime.clone() + y2.clone() - y1.clone()) % self.prime.clone();
        let exp = self.prime.clone() - BigUint::from_u8(2).unwrap();
        let dx_inv = dx.modpow(&exp, &self.prime);
        let lambda = dy.clone() * dx_inv % self.prime.clone();
        // Compute the quotient and carries of expr: λ * (x2 - x1) - y2 + y1.
        // expr can be positive or negative, but we need the quotient to be non-negative.
        let lambda_signed = BigInt::from_biguint(Sign::Plus, lambda.clone());
        let x1_signed = BigInt::from_biguint(Sign::Plus, x1.clone());
        let x2_signed = BigInt::from_biguint(Sign::Plus, x2.clone());
        let y1_signed = BigInt::from_biguint(Sign::Plus, y1.clone());
        let y2_signed = BigInt::from_biguint(Sign::Plus, y2.clone());
        let prime_signed = BigInt::from_biguint(Sign::Plus, self.prime.clone());
        let lambda_q_signed: BigInt =
            (lambda_signed.clone() * (x2_signed.clone() - x1_signed.clone()) - y2_signed
                + y1_signed.clone())
                / prime_signed.clone();
        let lambda_q_sign = lambda_q_signed.sign(); // TODO: should be in columns.
        let lambda_q_abs = bigint_abs(&lambda_q_signed);
        let lambda_q = to_canonical(&lambda_q_abs);
        for &q in lambda_q.limbs.iter() {
            range_check(self.limb_bits, q as usize);
        }
        // carries for expr: abs(λ * (x2 - x1) - y2 + y1) - λ_q * p
        let lambda_overflow = to_overflow_int(&lambda);
        let x1_overflow = to_overflow_int(&x1);
        let x2_overflow = to_overflow_int(&x2);
        let y1_overflow = to_overflow_int(&y1);
        let y2_overflow = to_overflow_int(&y2);
        let lambda_q_overflow = to_overflow_int(&lambda_q_abs);
        let prime_overflow = to_overflow_int(&self.prime);
        // TODO: expr should depends on sign
        let expr: OverflowInt<isize> =
            lambda_overflow.clone() * (x2_overflow.clone() - x1_overflow.clone()) - y2_overflow
                + y1_overflow.clone()
                - lambda_q_overflow * prime_overflow.clone();
        let lambda_carries = expr.calculate_carries(self.limb_bits);

        // ===== x3 =====
        // Compute x3: x3 = λ * λ - x1 - x2
        let x3 = (lambda.clone() * lambda.clone() + self.prime.clone() + self.prime.clone()
            - x1.clone()
            - x2.clone())
            % self.prime.clone();
        // Compute the quotient and carries of expr: λ * λ - x1 - x2 - x3
        let x3_signed = BigInt::from_biguint(Sign::Plus, x3.clone());
        let x3_q_signed = (lambda_signed.clone() * lambda_signed.clone()
            - x1_signed.clone()
            - x2_signed.clone()
            - x3_signed.clone())
            / prime_signed.clone();
        let x3_q_sign = x3_q_signed.sign();
        let x3_q_abs = bigint_abs(&x3_q_signed);
        let x3_q = to_canonical(&x3_q_abs);
        for &q in x3_q.limbs.iter() {
            range_check(self.limb_bits, q as usize);
        }
        // carries for expr: λ * λ - x1 - x2 - x3 - x3_q * p
        let x3_overflow = to_overflow_int(&x3);
        let x3_q_overflow = to_overflow_int(&x3_q_abs);
        let expr: OverflowInt<isize> = lambda_overflow.clone() * lambda_overflow.clone()
            - x1_overflow.clone()
            - x2_overflow.clone()
            - x3_overflow.clone()
            - x3_q_overflow * prime_overflow.clone();
        let x3_carries = expr.calculate_carries(self.limb_bits);

        // ===== y3 =====
        // Compute y3 and its carries: y3 = -λ * x3 - y1 + λ * x1.
        let y3 = ((self.prime.clone() + x1.clone() - x3.clone()) * lambda.clone()
            + self.prime.clone()
            - y1.clone())
            % self.prime.clone();
        // Compute the quotient and carries of expr: y3 + λ * x3 + y1 - λ * x1
        let y3_signed = BigInt::from_biguint(Sign::Plus, y3.clone());
        let y3_q_signed = (y3_signed + lambda_signed.clone() * x3_signed + y1_signed
            - lambda_signed * x1_signed)
            / prime_signed;
        let y3_q_sign = y3_q_signed.sign();
        let y3_q_abs = bigint_abs(&y3_q_signed);
        let y3_q = to_canonical(&y3_q_abs);
        for &q in y3_q.limbs.iter() {
            range_check(self.limb_bits, q as usize);
        }
        // carries for expr: y3 + λ * x3 + y1 - λ * x1 - y3_q * p
        let y3_overflow = to_overflow_int(&y3);
        let y3_q_overflow = to_overflow_int(&y3_q_abs);
        let expr: OverflowInt<isize> =
            y3_overflow + lambda_overflow.clone() * x3_overflow.clone() + y1_overflow.clone()
                - lambda_overflow.clone() * x1_overflow.clone()
                - y3_q_overflow * prime_overflow.clone();
        let y3_carries = expr.calculate_carries(self.limb_bits);

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
            lambda: vec_isize_to_f(lambda_overflow.limbs),
            lambda_check: CheckCarryModToZeroCols {
                carries: vec_isize_to_f(lambda_carries),
                quotient: vec_isize_to_f(lambda_q.limbs),
            },
            x3_check: CheckCarryModToZeroCols {
                carries: vec_isize_to_f(x3_carries),
                quotient: vec_isize_to_f(x3_q.limbs),
            },
            y3_check: CheckCarryModToZeroCols {
                carries: vec_isize_to_f(y3_carries),
                quotient: vec_isize_to_f(y3_q.limbs),
            },
        };

        EcAddCols { io, aux }
    }
}
