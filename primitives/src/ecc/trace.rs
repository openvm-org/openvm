use std::sync::Arc;

use num_bigint_dig::BigUint;
use p3_field::PrimeField64;

use super::{
    air::EccAir,
    columns::{EcAuxCols, EcAddCols, EcAddIoCols},
    utils::*,
    EcModularConfig, EcPoint,
};
use crate::{
    bigint::{
        check_carry_mod_to_zero::CheckCarryModToZeroCols,
        check_carry_to_zero::get_carry_max_abs_and_bits, utils::big_uint_mod_inverse,
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

        let config = EcModularConfig {
            prime: self.prime.clone(),
            num_limbs: self.num_limbs,
            limb_bits: self.limb_bits,
        };

        // ===== λ =====
        // Compute lambda: λ = (y2 - y1) / (x2 - x1).
        let dx = (&self.prime + &x2 - &x1) % &self.prime;
        let dy = (&self.prime + &y2 - &y1) % &self.prime;
        let dx_inv = big_uint_mod_inverse(&dx, &self.prime);
        let lambda: BigUint = (&dy * &dx_inv) % &self.prime;
        // Compute the quotient and carries of expr: λ * (x2 - x1) - y2 + y1.
        // expr can be positive or negative, so does q.
        let lambda_q_limbs = compute_lambda_q_limbs(&config, &x1, &x2, &y1, &y2, &lambda);
        for &q in lambda_q_limbs.iter() {
            range_checker.add_count((q + (1 << self.limb_bits)) as u32, self.limb_bits + 1);
        }
        let (lambda_carries, max_overflow_bits) =
            compute_lambda_carries(&config, &x1, &x2, &y1, &y2, &lambda, lambda_q_limbs.clone());
        let (carry_min_abs, carry_bits) =
            get_carry_max_abs_and_bits(max_overflow_bits, self.limb_bits);
        for &carry in lambda_carries.iter() {
            range_checker.add_count((carry + carry_min_abs as isize) as u32, carry_bits);
        }

        // ===== x3 =====
        // Compute x3: x3 = λ * λ - x1 - x2
        let x3: BigUint =
            (lambda.clone() * lambda.clone() + self.prime.clone() + self.prime.clone()
                - x1.clone()
                - x2.clone())
                % self.prime.clone();
        // Compute the quotient and carries of expr: λ * λ - x1 - x2 - x3
        let x3_q_limbs = compute_x3_q_limbs(&config, &x1, &x2, &x3, &lambda);
        for &q in x3_q_limbs.iter() {
            range_checker.add_count((q + (1 << self.limb_bits)) as u32, self.limb_bits + 1);
        }
        let (x3_carries, max_overflow_bits) =
            compute_x3_carries(&config, &x1, &x2, &x3, &lambda, x3_q_limbs.clone());
        let (carry_min_abs, carry_bits) =
            get_carry_max_abs_and_bits(max_overflow_bits, self.limb_bits);
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
        let y3_q_limbs = compute_y3_q_limbs(&config, &x1, &x3, &y1, &y3, &lambda);
        for &q in y3_q_limbs.iter() {
            range_checker.add_count((q + (1 << self.limb_bits)) as u32, self.limb_bits + 1);
        }
        // carries for expr: y3 + λ * x3 + y1 - λ * x1 - y3_q * p
        let (y3_carries, max_overflow_bits) =
            compute_y3_carries(&config, &x1, &x3, &y1, &y3, &lambda, y3_q_limbs.clone());
        let (carry_min_abs, carry_bits) =
            get_carry_max_abs_and_bits(max_overflow_bits, self.limb_bits);
        for &carry in y3_carries.iter() {
            range_checker.add_count((carry + carry_min_abs as isize) as u32, carry_bits);
        }

        let io = EcAddIoCols {
            p1: EcPoint {
                x: to_canonical_f(&x1, self.num_limbs),
                y: to_canonical_f(&y1, self.num_limbs),
            },
            p2: EcPoint {
                x: to_canonical_f(&x2, self.num_limbs),
                y: to_canonical_f(&y2, self.num_limbs),
            },
            p3: EcPoint {
                x: to_canonical_f(&x3, self.num_limbs),
                y: to_canonical_f(&y3, self.num_limbs),
            },
        };

        let aux = EcAuxCols {
            is_valid: F::one(),
            lambda: vec_isize_to_f(to_overflow_int(&lambda, self.num_limbs).limbs),
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
