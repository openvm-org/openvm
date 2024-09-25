use std::{
    cmp::max,
    ops::{Add, Mul, Sub},
};

use num_bigint_dig::BigUint;
use num_traits::One;
use stark_vm::modular_addsub::big_uint_mod_inverse;

use super::LIMB_BITS;

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
}

impl SymbolicExpr {
    // Maximum absolute positive and negative value of the expression.
    pub fn max_abs(&self, prime: &BigUint) -> (BigUint, BigUint) {
        match self {
            SymbolicExpr::Input(_) | SymbolicExpr::Var(_) => {
                (prime.clone() - BigUint::one(), BigUint::one())
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
        }
    }

    // If the expression is equal to q * p.
    // How many limbs does q have?
    // How many carry_limbs does it need to constrain expr - q * p = 0?
    pub fn constraint_limbs(&self, prime: &BigUint) -> (usize, usize) {
        let (max_pos_abs, max_neg_abs) = self.max_abs(prime);
        let max_abs = max(max_pos_abs, max_neg_abs);
        let max_abs = max_abs / prime;
        let expr_bits = max_abs.bits();
        let q_limbs = (expr_bits + LIMB_BITS - 1) / LIMB_BITS;

        let expr_limbs = (expr_bits + LIMB_BITS - 1) / LIMB_BITS;
        let p_bits = prime.bits();
        let p_limbs = (p_bits + LIMB_BITS - 1) / LIMB_BITS;
        let qp_limbs = q_limbs + p_limbs - 1;
        let carry_limbs = max(expr_limbs, qp_limbs);
        (q_limbs, carry_limbs)
    }

    // T will be BigInt, OverflowInt<isize>, OverflowInt<AB::Expr>
    pub fn evaluate<T>(&self, inputs: &[T], variables: &[T]) -> T
    where
        T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
    {
        match self {
            SymbolicExpr::Input(i) => inputs[*i].clone(),
            SymbolicExpr::Var(i) => variables[*i].clone(),
            SymbolicExpr::Add(lhs, rhs) => {
                lhs.evaluate(inputs, variables) + rhs.evaluate(inputs, variables)
            }
            SymbolicExpr::Sub(lhs, rhs) => {
                lhs.evaluate(inputs, variables) - rhs.evaluate(inputs, variables)
            }
            SymbolicExpr::Mul(lhs, rhs) => {
                lhs.evaluate(inputs, variables) * rhs.evaluate(inputs, variables)
            }
            SymbolicExpr::Div(_, _) => unreachable!(),
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
        }
    }
}
