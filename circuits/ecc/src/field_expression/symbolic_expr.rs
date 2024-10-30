use std::{
    cmp::{max, min},
    ops::{Add, Div, Mul, Sub},
};

use ax_circuit_primitives::bigint::{
    check_carry_to_zero::get_carry_max_abs_and_bits, utils::big_uint_mod_inverse, OverflowInt,
};
use num_bigint_dig::{BigInt, BigUint};
use num_traits::{FromPrimitive, One, Zero};
use p3_air::AirBuilder;
use p3_field::AbstractField;
use p3_util::log2_ceil_usize;

/// Example: If there are 4 inputs (x1, y1, x2, y2), and one intermediate variable lambda,
/// Mul(Var(0), Var(0)) - Input(0) - Input(2) =>
/// lambda * lambda - x1 - x2
#[derive(Clone, Debug, PartialEq)]
pub enum SymbolicExpr {
    Input(usize),
    Var(usize),
    Add(Box<SymbolicExpr>, Box<SymbolicExpr>),
    Sub(Box<SymbolicExpr>, Box<SymbolicExpr>),
    Mul(Box<SymbolicExpr>, Box<SymbolicExpr>),
    // Division is not allowed in "constraints", but can only be used in "computes"
    Div(Box<SymbolicExpr>, Box<SymbolicExpr>),
    // Add integer
    IntAdd(Box<SymbolicExpr>, isize),
    // Multiply each limb with an integer. For BigInt this is just scalar multiplication.
    IntMul(Box<SymbolicExpr>, isize),
    // Select one of the two expressions based on the flag.
    // The two expressions must have the same structure (number of limbs etc), e.g. a+b and a-b.
    Select(usize, Box<SymbolicExpr>, Box<SymbolicExpr>),
}

impl std::fmt::Display for SymbolicExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            SymbolicExpr::Input(i) => write!(f, "Input({})", i),
            SymbolicExpr::Var(i) => write!(f, "Var({})", i),
            SymbolicExpr::Add(lhs, rhs) => write!(f, "({} + {})", lhs, rhs),
            SymbolicExpr::Sub(lhs, rhs) => write!(f, "({} - {})", lhs, rhs),
            SymbolicExpr::Mul(lhs, rhs) => write!(f, "({} * {})", lhs, rhs),
            SymbolicExpr::Div(lhs, rhs) => write!(f, "({} / {})", lhs, rhs),
            SymbolicExpr::IntAdd(lhs, s) => write!(f, "({} + {})", lhs, s),
            SymbolicExpr::IntMul(lhs, s) => write!(f, "({} x {})", lhs, s),
            SymbolicExpr::Select(flag_id, lhs, rhs) => {
                write!(f, "(if {} then {} else {})", flag_id, lhs, rhs)
            }
        }
    }
}

impl Add for SymbolicExpr {
    type Output = SymbolicExpr;

    fn add(self, rhs: Self) -> Self::Output {
        SymbolicExpr::Add(Box::new(self), Box::new(rhs))
    }
}

impl Add<&SymbolicExpr> for SymbolicExpr {
    type Output = SymbolicExpr;

    fn add(self, rhs: &SymbolicExpr) -> Self::Output {
        SymbolicExpr::Add(Box::new(self), Box::new(rhs.clone()))
    }
}

impl Add for &SymbolicExpr {
    type Output = SymbolicExpr;

    fn add(self, rhs: &SymbolicExpr) -> Self::Output {
        SymbolicExpr::Add(Box::new(self.clone()), Box::new(rhs.clone()))
    }
}

impl Add<SymbolicExpr> for &SymbolicExpr {
    type Output = SymbolicExpr;

    fn add(self, rhs: SymbolicExpr) -> Self::Output {
        SymbolicExpr::Add(Box::new(self.clone()), Box::new(rhs))
    }
}

impl Sub for SymbolicExpr {
    type Output = SymbolicExpr;

    fn sub(self, rhs: Self) -> Self::Output {
        SymbolicExpr::Sub(Box::new(self), Box::new(rhs))
    }
}

impl Sub<&SymbolicExpr> for SymbolicExpr {
    type Output = SymbolicExpr;

    fn sub(self, rhs: &SymbolicExpr) -> Self::Output {
        SymbolicExpr::Sub(Box::new(self), Box::new(rhs.clone()))
    }
}

impl Sub for &SymbolicExpr {
    type Output = SymbolicExpr;

    fn sub(self, rhs: &SymbolicExpr) -> Self::Output {
        SymbolicExpr::Sub(Box::new(self.clone()), Box::new(rhs.clone()))
    }
}

impl Sub<SymbolicExpr> for &SymbolicExpr {
    type Output = SymbolicExpr;

    fn sub(self, rhs: SymbolicExpr) -> Self::Output {
        SymbolicExpr::Sub(Box::new(self.clone()), Box::new(rhs))
    }
}

impl Mul for SymbolicExpr {
    type Output = SymbolicExpr;

    fn mul(self, rhs: Self) -> Self::Output {
        SymbolicExpr::Mul(Box::new(self), Box::new(rhs))
    }
}

impl Mul<&SymbolicExpr> for SymbolicExpr {
    type Output = SymbolicExpr;

    fn mul(self, rhs: &SymbolicExpr) -> Self::Output {
        SymbolicExpr::Mul(Box::new(self), Box::new(rhs.clone()))
    }
}

impl Mul for &SymbolicExpr {
    type Output = SymbolicExpr;

    fn mul(self, rhs: &SymbolicExpr) -> Self::Output {
        SymbolicExpr::Mul(Box::new(self.clone()), Box::new(rhs.clone()))
    }
}

impl Mul<SymbolicExpr> for &SymbolicExpr {
    type Output = SymbolicExpr;

    fn mul(self, rhs: SymbolicExpr) -> Self::Output {
        SymbolicExpr::Mul(Box::new(self.clone()), Box::new(rhs))
    }
}

impl Div for SymbolicExpr {
    type Output = SymbolicExpr;

    fn div(self, rhs: Self) -> Self::Output {
        SymbolicExpr::Div(Box::new(self), Box::new(rhs))
    }
}

impl Div<&SymbolicExpr> for SymbolicExpr {
    type Output = SymbolicExpr;

    fn div(self, rhs: &SymbolicExpr) -> Self::Output {
        SymbolicExpr::Div(Box::new(self), Box::new(rhs.clone()))
    }
}

impl Div for &SymbolicExpr {
    type Output = SymbolicExpr;

    fn div(self, rhs: &SymbolicExpr) -> Self::Output {
        SymbolicExpr::Div(Box::new(self.clone()), Box::new(rhs.clone()))
    }
}

impl Div<SymbolicExpr> for &SymbolicExpr {
    type Output = SymbolicExpr;

    fn div(self, rhs: SymbolicExpr) -> Self::Output {
        SymbolicExpr::Div(Box::new(self.clone()), Box::new(rhs))
    }
}

impl SymbolicExpr {
    // Maximum absolute positive and negative value of the expression.
    // Needed in constraint_limbs to estimate the number of limbs of q.
    fn max_abs(&self, prime: &BigUint) -> (BigUint, BigUint) {
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
            SymbolicExpr::IntAdd(lhs, s) => {
                let (lhs_max_pos, lhs_max_neg) = lhs.max_abs(prime);
                let scalar = BigUint::from_usize(s.unsigned_abs()).unwrap();
                (lhs_max_pos + &scalar, lhs_max_neg + &scalar)
            }
            SymbolicExpr::IntMul(lhs, s) => {
                let (lhs_max_pos, lhs_max_neg) = lhs.max_abs(prime);
                let scalar = BigUint::from_usize(s.unsigned_abs()).unwrap();
                (lhs_max_pos * &scalar, lhs_max_neg * &scalar)
            }
            SymbolicExpr::Select(_, lhs, rhs) => {
                let (lhs_max_pos, lhs_max_neg) = lhs.max_abs(prime);
                let (rhs_max_pos, rhs_max_neg) = rhs.max_abs(prime);
                (max(lhs_max_pos, rhs_max_pos), max(lhs_max_neg, rhs_max_neg))
            }
        }
    }

    // Self should be a constraint expr.
    // This is already tracked in FieldVariable.
    // However in some cases (checking auto save in div), we need to know it from expr only.
    pub fn constraint_limb_max_abs(&self, limb_bits: usize, num_limbs: usize) -> usize {
        let canonical_limb_max_abs = (1 << limb_bits) - 1;
        match self {
            SymbolicExpr::Input(_) | SymbolicExpr::Var(_) => canonical_limb_max_abs,
            SymbolicExpr::Add(lhs, rhs) | SymbolicExpr::Sub(lhs, rhs) => {
                lhs.constraint_limb_max_abs(limb_bits, num_limbs)
                    + rhs.constraint_limb_max_abs(limb_bits, num_limbs)
            }
            SymbolicExpr::Mul(lhs, rhs) => {
                let left_num_limbs = lhs.expr_limbs(num_limbs);
                let right_num_limbs = rhs.expr_limbs(num_limbs);
                lhs.constraint_limb_max_abs(limb_bits, num_limbs)
                    * rhs.constraint_limb_max_abs(limb_bits, num_limbs)
                    * min(left_num_limbs, right_num_limbs)
            }
            SymbolicExpr::IntAdd(lhs, i) => {
                lhs.constraint_limb_max_abs(limb_bits, num_limbs) + i.unsigned_abs()
            }
            SymbolicExpr::IntMul(lhs, i) => {
                lhs.constraint_limb_max_abs(limb_bits, num_limbs) * i.unsigned_abs()
            }
            SymbolicExpr::Select(_, lhs, rhs) => max(
                lhs.constraint_limb_max_abs(limb_bits, num_limbs),
                rhs.constraint_limb_max_abs(limb_bits, num_limbs),
            ),
            SymbolicExpr::Div(_, _) => {
                unreachable!("should not have division when calling limb_max_abs")
            }
        }
    }

    // Self should be a constraint expr.
    // This returns the carry bits of self.expr - q * p.
    pub fn constraint_carry_bits_with_pq(
        &self,
        prime: &BigUint,
        limb_bits: usize,
        num_limbs: usize,
    ) -> usize {
        let without_pq = self.constraint_limb_max_abs(limb_bits, num_limbs);
        let (q_limbs, _) = self.constraint_limbs(prime, limb_bits, num_limbs);
        let canonical_limb_max_abs = (1 << limb_bits) - 1;
        let limb_max_abs =
            without_pq + canonical_limb_max_abs * canonical_limb_max_abs * min(q_limbs, num_limbs);
        let max_overflow_bits = log2_ceil_usize(limb_max_abs);
        let (_, carry_bits) = get_carry_max_abs_and_bits(max_overflow_bits, limb_bits);
        carry_bits
    }

    // Number of limbs to represent the expression.
    // num_limbs is the number of limbs of a canonical field element.
    pub fn expr_limbs(&self, num_limbs: usize) -> usize {
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
            SymbolicExpr::IntAdd(lhs, _) => lhs.expr_limbs(num_limbs),
            SymbolicExpr::IntMul(lhs, _) => lhs.expr_limbs(num_limbs),
            SymbolicExpr::Select(_, lhs, rhs) => {
                let left = lhs.expr_limbs(num_limbs);
                let right = rhs.expr_limbs(num_limbs);
                assert_eq!(left, right);
                left
            }
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
    pub fn evaluate_bigint(
        &self,
        inputs: &[BigInt],
        variables: &[BigInt],
        flags: &[bool],
    ) -> BigInt {
        match self {
            SymbolicExpr::IntAdd(lhs, s) => {
                lhs.evaluate_bigint(inputs, variables, flags) + BigInt::from_isize(*s).unwrap()
            }
            SymbolicExpr::IntMul(lhs, s) => {
                lhs.evaluate_bigint(inputs, variables, flags) * BigInt::from_isize(*s).unwrap()
            }
            SymbolicExpr::Input(i) => inputs[*i].clone(),
            SymbolicExpr::Var(i) => variables[*i].clone(),
            SymbolicExpr::Add(lhs, rhs) => {
                lhs.evaluate_bigint(inputs, variables, flags)
                    + rhs.evaluate_bigint(inputs, variables, flags)
            }
            SymbolicExpr::Sub(lhs, rhs) => {
                lhs.evaluate_bigint(inputs, variables, flags)
                    - rhs.evaluate_bigint(inputs, variables, flags)
            }
            SymbolicExpr::Mul(lhs, rhs) => {
                lhs.evaluate_bigint(inputs, variables, flags)
                    * rhs.evaluate_bigint(inputs, variables, flags)
            }
            SymbolicExpr::Select(flag_id, lhs, rhs) => {
                if flags[*flag_id] {
                    lhs.evaluate_bigint(inputs, variables, flags)
                } else {
                    rhs.evaluate_bigint(inputs, variables, flags)
                }
            }
            SymbolicExpr::Div(_, _) => unreachable!(), // Division is not allowed in constraints.
        }
    }

    // Used in trace gen to compute carries.
    pub fn evaluate_overflow_isize(
        &self,
        inputs: &[OverflowInt<isize>],
        variables: &[OverflowInt<isize>],
        flags: &[bool],
    ) -> OverflowInt<isize> {
        match self {
            SymbolicExpr::IntAdd(lhs, s) => {
                let mut left = lhs.evaluate_overflow_isize(inputs, variables, flags);
                left.limbs[0] += *s;
                left.limb_max_abs += s.unsigned_abs();
                left.max_overflow_bits = log2_ceil_usize(left.limb_max_abs);
                left
            }
            SymbolicExpr::IntMul(lhs, s) => {
                let mut left = lhs.evaluate_overflow_isize(inputs, variables, flags);
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
                lhs.evaluate_overflow_isize(inputs, variables, flags)
                    + rhs.evaluate_overflow_isize(inputs, variables, flags)
            }
            SymbolicExpr::Sub(lhs, rhs) => {
                lhs.evaluate_overflow_isize(inputs, variables, flags)
                    - rhs.evaluate_overflow_isize(inputs, variables, flags)
            }
            SymbolicExpr::Mul(lhs, rhs) => {
                lhs.evaluate_overflow_isize(inputs, variables, flags)
                    * rhs.evaluate_overflow_isize(inputs, variables, flags)
            }
            SymbolicExpr::Select(flag_id, lhs, rhs) => {
                if flags[*flag_id] {
                    lhs.evaluate_overflow_isize(inputs, variables, flags)
                } else {
                    rhs.evaluate_overflow_isize(inputs, variables, flags)
                }
            }
            SymbolicExpr::Div(_, _) => unreachable!(), // Division is not allowed in constraints.
        }
    }

    // Used in AIR eval.
    pub fn evaluate_overflow_expr<AB: AirBuilder>(
        &self,
        inputs: &[OverflowInt<AB::Expr>],
        variables: &[OverflowInt<AB::Expr>],
        flags: &[AB::Var],
    ) -> OverflowInt<AB::Expr> {
        match self {
            SymbolicExpr::IntAdd(lhs, s) => {
                let mut left = lhs.evaluate_overflow_expr::<AB>(inputs, variables, flags);
                let scalar = if *s >= 0 {
                    AB::Expr::from_canonical_usize(*s as usize)
                } else {
                    -AB::Expr::from_canonical_usize(s.unsigned_abs())
                };
                left.limbs[0] += scalar.clone();
                left.limb_max_abs += s.unsigned_abs();
                left.max_overflow_bits = log2_ceil_usize(left.limb_max_abs);
                left
            }
            SymbolicExpr::IntMul(lhs, s) => {
                let mut left = lhs.evaluate_overflow_expr::<AB>(inputs, variables, flags);
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
                lhs.evaluate_overflow_expr::<AB>(inputs, variables, flags)
                    + rhs.evaluate_overflow_expr::<AB>(inputs, variables, flags)
            }
            SymbolicExpr::Sub(lhs, rhs) => {
                lhs.evaluate_overflow_expr::<AB>(inputs, variables, flags)
                    - rhs.evaluate_overflow_expr::<AB>(inputs, variables, flags)
            }
            SymbolicExpr::Mul(lhs, rhs) => {
                lhs.evaluate_overflow_expr::<AB>(inputs, variables, flags)
                    * rhs.evaluate_overflow_expr::<AB>(inputs, variables, flags)
            }
            SymbolicExpr::Select(flag_id, lhs, rhs) => {
                let left = lhs.evaluate_overflow_expr::<AB>(inputs, variables, flags);
                let right = rhs.evaluate_overflow_expr::<AB>(inputs, variables, flags);
                assert_eq!(left.limbs.len(), right.limbs.len());
                let flag = flags[*flag_id];
                let mut res = vec![];
                for i in 0..left.limbs.len() {
                    res.push(
                        left.limbs[i].clone() * flag.into()
                            + right.limbs[i].clone() * (AB::Expr::one() - flag.into()),
                    );
                }
                OverflowInt {
                    limbs: res,
                    limb_max_abs: left.limb_max_abs,
                    max_overflow_bits: left.max_overflow_bits,
                }
            }
            SymbolicExpr::Div(_, _) => unreachable!(), // Division is not allowed in constraints.
        }
    }

    // Result will be within [0, prime).
    pub fn compute(
        &self,
        inputs: &[BigUint],
        variables: &[BigUint],
        flags: &[bool],
        prime: &BigUint,
    ) -> BigUint {
        match self {
            SymbolicExpr::Input(i) => inputs[*i].clone(),
            SymbolicExpr::Var(i) => variables[*i].clone(),
            SymbolicExpr::Add(lhs, rhs) => {
                (lhs.compute(inputs, variables, flags, prime)
                    + rhs.compute(inputs, variables, flags, prime))
                    % prime
            }
            SymbolicExpr::Sub(lhs, rhs) => {
                (prime + lhs.compute(inputs, variables, flags, prime)
                    - rhs.compute(inputs, variables, flags, prime))
                    % prime
            }
            SymbolicExpr::Mul(lhs, rhs) => {
                (lhs.compute(inputs, variables, flags, prime)
                    * rhs.compute(inputs, variables, flags, prime))
                    % prime
            }
            SymbolicExpr::Div(lhs, rhs) => {
                let left = lhs.compute(inputs, variables, flags, prime);
                let right = rhs.compute(inputs, variables, flags, prime);
                let right_inv = big_uint_mod_inverse(&right, prime);
                (left * right_inv) % prime
            }
            SymbolicExpr::IntAdd(lhs, s) => {
                let left = lhs.compute(inputs, variables, flags, prime);
                let right = if *s >= 0 {
                    BigUint::from_usize(*s as usize).unwrap()
                } else {
                    prime - BigUint::from_usize(s.unsigned_abs()).unwrap()
                };
                (left + right) % prime
            }
            SymbolicExpr::IntMul(lhs, s) => {
                let left = lhs.compute(inputs, variables, flags, prime);
                let right = if *s >= 0 {
                    BigUint::from_usize(*s as usize).unwrap()
                } else {
                    prime - BigUint::from_usize(s.unsigned_abs()).unwrap()
                };
                (left * right) % prime
            }
            SymbolicExpr::Select(flag_id, lhs, rhs) => {
                if flags[*flag_id] {
                    lhs.compute(inputs, variables, flags, prime)
                } else {
                    rhs.compute(inputs, variables, flags, prime)
                }
            }
        }
    }
}
