use std::cmp::max;

use afs_primitives::bigint::OverflowInt;
use num_bigint_dig::{BigInt, BigUint};
use num_traits::{FromPrimitive, One, Zero};
use p3_air::AirBuilder;
use p3_field::AbstractField;
use p3_util::log2_ceil_usize;
use stark_vm::modular_addsub::big_uint_mod_inverse;

/// Example: If there are 4 inputs (x1, y1, x2, y2), and one intermediate variable lambda,
/// Mul(Var(0), Var(0)) - Input(0) - Input(2) =>
/// lambda * lambda - x1 - x2
#[derive(Clone, Debug)]
pub enum SymbolicExpr {
    Input(usize),
    Var(usize),
    Add(Box<SymbolicExpr>, Box<SymbolicExpr>),
    Sub(Box<SymbolicExpr>, Box<SymbolicExpr>),
    Mul(Box<SymbolicExpr>, Box<SymbolicExpr>),
    // Division is not allowed in "constraints", but can only be used in "computes"
    Div(Box<SymbolicExpr>, Box<SymbolicExpr>),
    // Multiply each limb with an integer. For BigInt this is just scalar multiplication.
    IntMul(Box<SymbolicExpr>, isize),
}

impl SymbolicExpr {
    // Maximum absolute positive and negative value of the expression.
    pub fn max_abs(&self, prime: &BigUint) -> (BigUint, BigUint) {
        match self {
            SymbolicExpr::Input(_) | SymbolicExpr::Var(_) => {
                // Input and variable are field elements so are in [0, p)
                (prime.clone() - BigUint::one(), BigUint::zero())
            }
            SymbolicExpr::Add(lhs, rhs) => {
                let (lhs_max_pos, lhs_max_neg) = lhs.max_abs(prime);
                let (rhs_max_pos, rhs_max_neg) = rhs.max_abs(prime);
                (lhs_max_pos + rhs_max_pos, lhs_max_neg + rhs_max_neg)
            }
            SymbolicExpr::Sub(lhs, rhs) => {
                let (lhs_max_pos, lhs_max_neg) = lhs.max_abs(prime);
                let (rhs_max_pos, rhs_max_neg) = rhs.max_abs(prime);
                (lhs_max_pos + rhs_max_neg, lhs_max_neg + rhs_max_pos)
            }
            SymbolicExpr::Mul(lhs, rhs) => {
                let (lhs_max_pos, lhs_max_neg) = lhs.max_abs(prime);
                let (rhs_max_pos, rhs_max_neg) = rhs.max_abs(prime);
                (
                    max(&lhs_max_pos * &rhs_max_pos, &lhs_max_neg * &rhs_max_neg),
                    max(&lhs_max_pos * &rhs_max_neg, &lhs_max_neg * &rhs_max_pos),
                )
            }
            SymbolicExpr::Div(_, _) => {
                // Should not have division in expression when calling this.
                unreachable!()
            }
            SymbolicExpr::IntMul(lhs, s) => {
                let (lhs_max_pos, lhs_max_neg) = lhs.max_abs(prime);
                let scalar = BigUint::from_usize(s.unsigned_abs()).unwrap();
                (lhs_max_pos * &scalar, lhs_max_neg * &scalar)
            }
        }
    }

    fn expr_limbs(&self, num_limbs: usize) -> usize {
        match self {
            SymbolicExpr::Input(_) | SymbolicExpr::Var(_) => num_limbs,
            SymbolicExpr::Add(lhs, rhs) | SymbolicExpr::Sub(lhs, rhs) => {
                max(lhs.expr_limbs(num_limbs), rhs.expr_limbs(num_limbs))
            }
            SymbolicExpr::Mul(lhs, rhs) => {
                lhs.expr_limbs(num_limbs) + rhs.expr_limbs(num_limbs) - 1
            }
            SymbolicExpr::Div(_, _) => {
                unimplemented!()
            }
            SymbolicExpr::IntMul(lhs, _) => lhs.expr_limbs(num_limbs),
        }
    }

    // If the expression is equal to q * p.
    // How many limbs does q have?
    // How many carry_limbs does it need to constrain expr - q * p = 0?
    pub fn constraint_limbs(
        &self,
        prime: &BigUint,
        limb_bits: usize,
        num_limbs: usize,
    ) -> (usize, usize) {
        let (max_pos_abs, max_neg_abs) = self.max_abs(prime);
        let max_abs = max(max_pos_abs, max_neg_abs);
        let max_q_abs = (&max_abs + prime - BigUint::one()) / prime;
        let q_bits = max_q_abs.bits();
        let q_limbs = q_bits.div_ceil(limb_bits);
        let p_limbs = prime.bits().div_ceil(limb_bits);
        let qp_limbs = q_limbs + p_limbs - 1;

        let expr_limbs = self.expr_limbs(num_limbs);
        let carry_limbs = max(expr_limbs, qp_limbs);
        (q_limbs, carry_limbs)
    }

    // Used in trace gen to compute q.
    pub fn evaluate_bigint(&self, inputs: &[BigInt], variables: &[BigInt]) -> BigInt {
        match self {
            SymbolicExpr::IntMul(lhs, s) => {
                lhs.evaluate_bigint(inputs, variables) * BigInt::from_isize(*s).unwrap()
            }
            SymbolicExpr::Input(i) => inputs[*i].clone(),
            SymbolicExpr::Var(i) => variables[*i].clone(),
            SymbolicExpr::Add(lhs, rhs) => {
                lhs.evaluate_bigint(inputs, variables) + rhs.evaluate_bigint(inputs, variables)
            }
            SymbolicExpr::Sub(lhs, rhs) => {
                lhs.evaluate_bigint(inputs, variables) - rhs.evaluate_bigint(inputs, variables)
            }
            SymbolicExpr::Mul(lhs, rhs) => {
                lhs.evaluate_bigint(inputs, variables) * rhs.evaluate_bigint(inputs, variables)
            }
            SymbolicExpr::Div(_, _) => unreachable!(), // Division is not allowed in constraints.
        }
    }

    // Used in trace gen to compute carries.
    pub fn evaluate_overflow_isize(
        &self,
        inputs: &[OverflowInt<isize>],
        variables: &[OverflowInt<isize>],
    ) -> OverflowInt<isize> {
        match self {
            SymbolicExpr::IntMul(lhs, s) => {
                let mut left = lhs.evaluate_overflow_isize(inputs, variables);
                for limb in left.limbs.iter_mut() {
                    *limb *= *s;
                }
                left.limb_max_abs *= s.unsigned_abs();
                left.max_overflow_bits = log2_ceil_usize(left.limb_max_abs);
                left
            }
            SymbolicExpr::Input(i) => inputs[*i].clone(),
            SymbolicExpr::Var(i) => variables[*i].clone(),
            SymbolicExpr::Add(lhs, rhs) => {
                lhs.evaluate_overflow_isize(inputs, variables)
                    + rhs.evaluate_overflow_isize(inputs, variables)
            }
            SymbolicExpr::Sub(lhs, rhs) => {
                lhs.evaluate_overflow_isize(inputs, variables)
                    - rhs.evaluate_overflow_isize(inputs, variables)
            }
            SymbolicExpr::Mul(lhs, rhs) => {
                lhs.evaluate_overflow_isize(inputs, variables)
                    * rhs.evaluate_overflow_isize(inputs, variables)
            }
            SymbolicExpr::Div(_, _) => unreachable!(), // Division is not allowed in constraints.
        }
    }

    // Used in AIR eval.
    pub fn evaluate_overflow_expr<AB: AirBuilder>(
        &self,
        inputs: &[OverflowInt<AB::Expr>],
        variables: &[OverflowInt<AB::Expr>],
    ) -> OverflowInt<AB::Expr> {
        match self {
            SymbolicExpr::IntMul(lhs, s) => {
                let mut left = lhs.evaluate_overflow_expr::<AB>(inputs, variables);
                let scalar = if *s >= 0 {
                    AB::Expr::from_canonical_usize(*s as usize)
                } else {
                    -AB::Expr::from_canonical_usize(s.unsigned_abs())
                };
                for limb in left.limbs.iter_mut() {
                    *limb *= scalar.clone();
                }
                left.limb_max_abs *= s.unsigned_abs();
                left.max_overflow_bits = log2_ceil_usize(left.limb_max_abs);
                left
            }
            SymbolicExpr::Input(i) => inputs[*i].clone(),
            SymbolicExpr::Var(i) => variables[*i].clone(),
            SymbolicExpr::Add(lhs, rhs) => {
                lhs.evaluate_overflow_expr::<AB>(inputs, variables)
                    + rhs.evaluate_overflow_expr::<AB>(inputs, variables)
            }
            SymbolicExpr::Sub(lhs, rhs) => {
                lhs.evaluate_overflow_expr::<AB>(inputs, variables)
                    - rhs.evaluate_overflow_expr::<AB>(inputs, variables)
            }
            SymbolicExpr::Mul(lhs, rhs) => {
                lhs.evaluate_overflow_expr::<AB>(inputs, variables)
                    * rhs.evaluate_overflow_expr::<AB>(inputs, variables)
            }
            SymbolicExpr::Div(_, _) => unreachable!(), // Division is not allowed in constraints.
        }
    }

    // Result will be within [0, prime).
    pub fn compute(&self, inputs: &[BigUint], variables: &[BigUint], prime: &BigUint) -> BigUint {
        match self {
            SymbolicExpr::Input(i) => inputs[*i].clone(),
            SymbolicExpr::Var(i) => variables[*i].clone(),
            SymbolicExpr::Add(lhs, rhs) => {
                (lhs.compute(inputs, variables, prime) + rhs.compute(inputs, variables, prime))
                    % prime
            }
            SymbolicExpr::Sub(lhs, rhs) => {
                (prime + lhs.compute(inputs, variables, prime)
                    - rhs.compute(inputs, variables, prime))
                    % prime
            }
            SymbolicExpr::Mul(lhs, rhs) => {
                (lhs.compute(inputs, variables, prime) * rhs.compute(inputs, variables, prime))
                    % prime
            }
            SymbolicExpr::Div(lhs, rhs) => {
                let left = lhs.compute(inputs, variables, prime);
                let right = rhs.compute(inputs, variables, prime);
                let right_inv = big_uint_mod_inverse(&right, prime);
                (left * right_inv) % prime
            }
            SymbolicExpr::IntMul(lhs, s) => {
                let left = lhs.compute(inputs, variables, prime);
                let right = if *s >= 0 {
                    BigUint::from_usize(*s as usize).unwrap()
                } else {
                    prime - BigUint::from_usize(s.unsigned_abs()).unwrap()
                };
                (left * right) % prime
            }
        }
    }
}
