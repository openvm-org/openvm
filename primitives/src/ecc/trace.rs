use std::sync::Arc;

use num_bigint_dig::{BigInt, BigUint, Sign};
use p3_field::PrimeField64;

use super::{
    air::EccAir,
    columns::{EcAddAuxCols, EcAddCols, EcAddIoCols},
    EcPoint,
};
use crate::{
    bigint::{
        check_carry_mod_to_zero::CheckCarryModToZeroCols,
        check_carry_to_zero::get_carry_max_abs_and_bits,
        utils::{big_int_to_num_limbs, big_uint_mod_inverse},
        CanonicalUint, DefaultLimbConfig, OverflowInt,
    },
    sub_chip::LocalTraceInstructions,
    var_range::VariableRangeCheckerChip,
};

impl<F: PrimeField64> LocalTraceInstructions<F> for EccAir {
    type LocalInput = (
        (BigUint, BigUint),
        (BigUint, BigUint),
        Arc<VariableRangeCheckerChip>,
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
        let to_overflow_q = |q_limbs: Vec<isize>| OverflowInt {
            limbs: q_limbs,
            max_overflow_bits: self.limb_bits + 1,
            limb_max_abs: (1 << self.limb_bits),
        };

        // ===== λ =====
        // Compute lambda: λ = (y2 - y1) / (x2 - x1).
        let dx = (self.prime.clone() + x2.clone() - x1.clone()) % self.prime.clone();
        let dy = (self.prime.clone() + y2.clone() - y1.clone()) % self.prime.clone();
        let dx_inv = big_uint_mod_inverse(&dx, &self.prime);
        let lambda = dy.clone() * dx_inv % self.prime.clone();
        // Compute the quotient and carries of expr: λ * (x2 - x1) - y2 + y1.
        // expr can be positive or negative, so does q.
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
        let lambda_q_limbs: Vec<isize> =
            big_int_to_num_limbs(lambda_q_signed, self.limb_bits, self.num_limbs);
        for &q in lambda_q_limbs.iter() {
            range_checker.add_count((q + (1 << self.limb_bits)) as u32, self.limb_bits + 1);
        }
        // carries for expr: abs(λ * (x2 - x1) - y2 + y1) - λ_q * p
        let lambda_overflow = to_overflow_int(&lambda);
        let x1_overflow = to_overflow_int(&x1);
        let x2_overflow = to_overflow_int(&x2);
        let y1_overflow = to_overflow_int(&y1);
        let y2_overflow = to_overflow_int(&y2);
        let lambda_q_overflow = to_overflow_q(lambda_q_limbs.clone());
        let prime_overflow = to_overflow_int(&self.prime);
        // Taking abs of λ * (x2 - x1) - y2 + y1
        let expr = lambda_overflow.clone() * (x2_overflow.clone() - x1_overflow.clone())
            - y2_overflow
            + y1_overflow.clone();
        let expr = expr - lambda_q_overflow * prime_overflow.clone();
        let lambda_carries = expr.calculate_carries(self.limb_bits);
        let (carry_min_abs, carry_bits) =
            get_carry_max_abs_and_bits(expr.max_overflow_bits, self.limb_bits);
        for &carry in lambda_carries.iter() {
            range_checker.add_count((carry + carry_min_abs as isize) as u32, carry_bits);
        }

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
        let x3_q_limbs: Vec<isize> =
            big_int_to_num_limbs(x3_q_signed, self.limb_bits, self.num_limbs);
        for &q in x3_q_limbs.iter() {
            range_checker.add_count((q + (1 << self.limb_bits)) as u32, self.limb_bits + 1);
        }
        // carries for expr: λ * λ - x1 - x2 - x3 - x3_q * p
        let x3_overflow = to_overflow_int(&x3);
        let x3_q_overflow = to_overflow_q(x3_q_limbs.clone());
        let expr: OverflowInt<isize> = lambda_overflow.clone() * lambda_overflow.clone()
            - x1_overflow.clone()
            - x2_overflow.clone()
            - x3_overflow.clone()
            - x3_q_overflow * prime_overflow.clone();
        let x3_carries = expr.calculate_carries(self.limb_bits);
        let (carry_min_abs, carry_bits) =
            get_carry_max_abs_and_bits(expr.max_overflow_bits, self.limb_bits);
        for &carry in x3_carries.iter() {
            range_checker.add_count((carry + carry_min_abs as isize) as u32, carry_bits);
        }

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
        let y3_q_limbs: Vec<isize> =
            big_int_to_num_limbs(y3_q_signed, self.limb_bits, self.num_limbs);
        for &q in y3_q_limbs.iter() {
            range_checker.add_count((q + (1 << self.limb_bits)) as u32, self.limb_bits + 1);
        }
        // carries for expr: y3 + λ * x3 + y1 - λ * x1 - y3_q * p
        let y3_overflow = to_overflow_int(&y3);
        let y3_q_overflow = to_overflow_q(y3_q_limbs.clone());
        let expr: OverflowInt<isize> =
            y3_overflow + lambda_overflow.clone() * x3_overflow.clone() + y1_overflow.clone()
                - lambda_overflow.clone() * x1_overflow.clone()
                - y3_q_overflow * prime_overflow.clone();
        let y3_carries = expr.calculate_carries(self.limb_bits);
        let (carry_min_abs, carry_bits) =
            get_carry_max_abs_and_bits(expr.max_overflow_bits, self.limb_bits);
        for &carry in y3_carries.iter() {
            range_checker.add_count((carry + carry_min_abs as isize) as u32, carry_bits);
        }

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
            is_valid: F::one(),
            lambda: vec_isize_to_f(lambda_overflow.limbs),
            lambda_check: CheckCarryModToZeroCols {
                carries: vec_isize_to_f(lambda_carries),
                quotient: vec_isize_to_f(lambda_q_limbs),
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
