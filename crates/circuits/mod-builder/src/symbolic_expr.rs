use std::{
    cmp::{max, min},
    convert::identity,
    iter::repeat,
    ops::{Add, Div, Mul, Sub},
};

use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{FromPrimitive, One, Zero};
use openvm_circuit_primitives::bigint::{
    check_carry_to_zero::get_carry_max_abs_and_bits, OverflowInt,
};
use openvm_stark_backend::{p3_air::AirBuilder, p3_field::FieldAlgebra, p3_util::log2_ceil_usize};

/// Example: If there are 4 inputs (x1, y1, x2, y2), and one intermediate variable lambda,
/// Mul(Var(0), Var(0)) - Input(0) - Input(2) =>
/// lambda * lambda - x1 - x2
#[derive(Clone, Debug, PartialEq)]
pub enum SymbolicExpr {
    Input(usize),
    Var(usize),
    Const(usize, BigUint, usize), // (index, value, number of limbs)
    Add(Box<SymbolicExpr>, Box<SymbolicExpr>),
    Sub(Box<SymbolicExpr>, Box<SymbolicExpr>),
    Mul(Box<SymbolicExpr>, Box<SymbolicExpr>),
    // Division is not allowed in "constraints", but can only be used in "computes"
    // Note that division by zero in "computes" will panic.
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
            SymbolicExpr::Input(i) => write!(f, "Input_{}", i),
            SymbolicExpr::Var(i) => write!(f, "Var_{}", i),
            SymbolicExpr::Const(i, _, _) => write!(f, "Const_{}", i),
            SymbolicExpr::Add(lhs, rhs) => write!(f, "({} + {})", lhs, rhs),
            SymbolicExpr::Sub(lhs, rhs) => write!(f, "({} - {})", lhs, rhs),
            SymbolicExpr::Mul(lhs, rhs) => write!(f, "{} * {}", lhs, rhs),
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

// Note that division by zero will panic.
impl Div for SymbolicExpr {
    type Output = SymbolicExpr;

    fn div(self, rhs: Self) -> Self::Output {
        SymbolicExpr::Div(Box::new(self), Box::new(rhs))
    }
}

// Note that division by zero will panic.
impl Div<&SymbolicExpr> for SymbolicExpr {
    type Output = SymbolicExpr;

    fn div(self, rhs: &SymbolicExpr) -> Self::Output {
        SymbolicExpr::Div(Box::new(self), Box::new(rhs.clone()))
    }
}

// Note that division by zero will panic.
impl Div for &SymbolicExpr {
    type Output = SymbolicExpr;

    fn div(self, rhs: &SymbolicExpr) -> Self::Output {
        SymbolicExpr::Div(Box::new(self.clone()), Box::new(rhs.clone()))
    }
}

// Note that division by zero will panic.
impl Div<SymbolicExpr> for &SymbolicExpr {
    type Output = SymbolicExpr;

    fn div(self, rhs: SymbolicExpr) -> Self::Output {
        SymbolicExpr::Div(Box::new(self.clone()), Box::new(rhs))
    }
}

impl SymbolicExpr {
    /// Returns maximum absolute positive and negative value of the expression.
    /// That is, if `(r, l) = expr.max_abs(p)` then `l,r >= 0` and `-l <= expr <= r`.
    /// Needed in `constraint_limbs` to estimate the number of limbs of q.
    ///
    /// It is assumed that any `Input` or `Var` is a non-negative big integer with value
    /// in the range `[0, proper_max]`.
    fn max_abs(&self, proper_max: &BigUint) -> (BigUint, BigUint) {
        match self {
            SymbolicExpr::Input(_) | SymbolicExpr::Var(_) => (proper_max.clone(), BigUint::zero()),
            SymbolicExpr::Const(_, val, _) => (val.clone(), BigUint::zero()),
            SymbolicExpr::Add(lhs, rhs) => {
                let (lhs_max_pos, lhs_max_neg) = lhs.max_abs(proper_max);
                let (rhs_max_pos, rhs_max_neg) = rhs.max_abs(proper_max);
                (lhs_max_pos + rhs_max_pos, lhs_max_neg + rhs_max_neg)
            }
            SymbolicExpr::Sub(lhs, rhs) => {
                let (lhs_max_pos, lhs_max_neg) = lhs.max_abs(proper_max);
                let (rhs_max_pos, rhs_max_neg) = rhs.max_abs(proper_max);
                (lhs_max_pos + rhs_max_neg, lhs_max_neg + rhs_max_pos)
            }
            SymbolicExpr::Mul(lhs, rhs) => {
                let (lhs_max_pos, lhs_max_neg) = lhs.max_abs(proper_max);
                let (rhs_max_pos, rhs_max_neg) = rhs.max_abs(proper_max);
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
                let (lhs_max_pos, lhs_max_neg) = lhs.max_abs(proper_max);
                let scalar = BigUint::from_usize(s.unsigned_abs()).unwrap();
                // Optimization opportunity: since `s` is a constant, we can likely do better than
                // this bound.
                (lhs_max_pos + &scalar, lhs_max_neg + &scalar)
            }
            SymbolicExpr::IntMul(lhs, s) => {
                let (lhs_max_pos, lhs_max_neg) = lhs.max_abs(proper_max);
                let scalar = BigUint::from_usize(s.unsigned_abs()).unwrap();
                if *s < 0 {
                    (lhs_max_neg * &scalar, lhs_max_pos * &scalar)
                } else {
                    (lhs_max_pos * &scalar, lhs_max_neg * &scalar)
                }
            }
            SymbolicExpr::Select(_, lhs, rhs) => {
                let (lhs_max_pos, lhs_max_neg) = lhs.max_abs(proper_max);
                let (rhs_max_pos, rhs_max_neg) = rhs.max_abs(proper_max);
                (max(lhs_max_pos, rhs_max_pos), max(lhs_max_neg, rhs_max_neg))
            }
        }
    }

    /// Returns the maximum possible size, in bits, of each limb in `self.expr`.
    /// This is already tracked in `FieldVariable`. However when auto saving in
    /// `FieldVariable::div`, we need to know it from the `SymbolicExpr` only.
    /// self should be a constraint expr.
    pub fn constraint_limb_max_abs(&self, limb_bits: usize, num_limbs: usize) -> usize {
        let canonical_limb_max_abs = (1 << limb_bits) - 1;
        match self {
            SymbolicExpr::Input(_) | SymbolicExpr::Var(_) | SymbolicExpr::Const(_, _, _) => {
                canonical_limb_max_abs
            }
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

    /// Returns the maximum possible size, in bits, of each carry in `self.expr - q * p`.
    /// self should be a constraint expr.
    ///
    /// The cached value `proper_max` should equal `2^{limb_bits * num_limbs} - 1`.
    pub fn constraint_carry_bits_with_pq(
        &self,
        prime: &BigUint,
        limb_bits: usize,
        num_limbs: usize,
        proper_max: &BigUint,
    ) -> usize {
        let without_pq = self.constraint_limb_max_abs(limb_bits, num_limbs);
        let (q_limbs, _) = self.constraint_limbs(prime, limb_bits, num_limbs, proper_max);
        let canonical_limb_max_abs = (1 << limb_bits) - 1;
        let limb_max_abs =
            without_pq + canonical_limb_max_abs * canonical_limb_max_abs * min(q_limbs, num_limbs);
        let max_overflow_bits = log2_ceil_usize(limb_max_abs);
        let (_, carry_bits) = get_carry_max_abs_and_bits(max_overflow_bits, limb_bits);
        carry_bits
    }

    /// Returns the number of limbs needed to represent the expression.
    /// The parameter `num_limbs` is the number of limbs of a canonical field element.
    pub fn expr_limbs(&self, num_limbs: usize) -> usize {
        match self {
            SymbolicExpr::Input(_) | SymbolicExpr::Var(_) => num_limbs,
            SymbolicExpr::Const(_, _, limbs) => *limbs,
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

    /// Let `q` be such that `self.expr = q * p`.
    /// Returns (q_limbs, carry_limbs) where q_limbs is the number of limbs in q
    /// and carry_limbs is the number of limbs in the carry of the constraint self.expr - q * p = 0.
    /// self should be a constraint expression.
    ///
    /// The cached value `proper_max` should equal `2^{limb_bits * num_limbs} - 1`.
    pub fn constraint_limbs(
        &self,
        prime: &BigUint,
        limb_bits: usize,
        num_limbs: usize,
        proper_max: &BigUint,
    ) -> (usize, usize) {
        let (max_pos_abs, max_neg_abs) = self.max_abs(proper_max);
        let max_abs = max(max_pos_abs, max_neg_abs);
        let max_q_abs = (&max_abs + prime - BigUint::one()) / prime;
        let q_bits = max_q_abs.bits() as usize;
        let p_bits = prime.bits() as usize;
        let q_limbs = q_bits.div_ceil(limb_bits);
        // Attention! This must match with prime_overflow in `FieldExpr::generate_subrow`
        let p_limbs = p_bits.div_ceil(limb_bits);
        let qp_limbs = q_limbs + p_limbs - 1;

        let expr_limbs = self.expr_limbs(num_limbs);
        let carry_limbs = max(expr_limbs, qp_limbs);
        (q_limbs, carry_limbs)
    }

    /// Used in trace gen to compute `q``.
    /// self should be a constraint expression.
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
            SymbolicExpr::Const(_, val, _) => {
                if val.is_zero() {
                    BigInt::zero()
                } else {
                    BigInt::from_biguint(Sign::Plus, val.clone())
                }
            }
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

    /// Used in trace gen to compute carries.
    /// self should be a constraint expression.
    pub fn evaluate_overflow_isize(
        &self,
        inputs: &[OverflowInt<isize>],
        variables: &[OverflowInt<isize>],
        constants: &[OverflowInt<isize>],
        flags: &[bool],
    ) -> OverflowInt<isize> {
        match self {
            SymbolicExpr::IntAdd(lhs, s) => {
                let left = lhs.evaluate_overflow_isize(inputs, variables, constants, flags);
                left.int_add(*s, identity)
            }
            SymbolicExpr::IntMul(lhs, s) => {
                let left = lhs.evaluate_overflow_isize(inputs, variables, constants, flags);
                left.int_mul(*s, identity)
            }
            SymbolicExpr::Input(i) => inputs[*i].clone(),
            SymbolicExpr::Var(i) => variables[*i].clone(),
            SymbolicExpr::Const(i, _, _) => constants[*i].clone(),
            SymbolicExpr::Add(lhs, rhs) => {
                lhs.evaluate_overflow_isize(inputs, variables, constants, flags)
                    + rhs.evaluate_overflow_isize(inputs, variables, constants, flags)
            }
            SymbolicExpr::Sub(lhs, rhs) => {
                lhs.evaluate_overflow_isize(inputs, variables, constants, flags)
                    - rhs.evaluate_overflow_isize(inputs, variables, constants, flags)
            }
            SymbolicExpr::Mul(lhs, rhs) => {
                lhs.evaluate_overflow_isize(inputs, variables, constants, flags)
                    * rhs.evaluate_overflow_isize(inputs, variables, constants, flags)
            }
            SymbolicExpr::Select(flag_id, lhs, rhs) => {
                let left = lhs.evaluate_overflow_isize(inputs, variables, constants, flags);
                let right = rhs.evaluate_overflow_isize(inputs, variables, constants, flags);
                let num_limbs = max(left.num_limbs(), right.num_limbs());

                let res = if flags[*flag_id] {
                    left.limbs().to_vec()
                } else {
                    right.limbs().to_vec()
                };
                let res = res.into_iter().chain(repeat(0)).take(num_limbs).collect();

                OverflowInt::from_computed_limbs(
                    res,
                    max(left.limb_max_abs(), right.limb_max_abs()),
                    max(left.max_overflow_bits(), right.max_overflow_bits()),
                )
            }
            SymbolicExpr::Div(_, _) => unreachable!(), // Division is not allowed in constraints.
        }
    }

    fn isize_to_expr<AB: AirBuilder>(s: isize) -> AB::Expr {
        if s >= 0 {
            AB::Expr::from_canonical_usize(s as usize)
        } else {
            -AB::Expr::from_canonical_usize(s.unsigned_abs())
        }
    }

    /// Used in AIR eval.
    /// self should be a constraint expression.
    pub fn evaluate_overflow_expr<AB: AirBuilder>(
        &self,
        inputs: &[OverflowInt<AB::Expr>],
        variables: &[OverflowInt<AB::Expr>],
        constants: &[OverflowInt<AB::Expr>],
        flags: &[AB::Var],
    ) -> OverflowInt<AB::Expr> {
        match self {
            SymbolicExpr::IntAdd(lhs, s) => {
                let left = lhs.evaluate_overflow_expr::<AB>(inputs, variables, constants, flags);
                left.int_add(*s, Self::isize_to_expr::<AB>)
            }
            SymbolicExpr::IntMul(lhs, s) => {
                let left = lhs.evaluate_overflow_expr::<AB>(inputs, variables, constants, flags);
                left.int_mul(*s, Self::isize_to_expr::<AB>)
            }
            SymbolicExpr::Input(i) => inputs[*i].clone(),
            SymbolicExpr::Var(i) => variables[*i].clone(),
            SymbolicExpr::Const(i, _, _) => constants[*i].clone(),
            SymbolicExpr::Add(lhs, rhs) => {
                lhs.evaluate_overflow_expr::<AB>(inputs, variables, constants, flags)
                    + rhs.evaluate_overflow_expr::<AB>(inputs, variables, constants, flags)
            }
            SymbolicExpr::Sub(lhs, rhs) => {
                lhs.evaluate_overflow_expr::<AB>(inputs, variables, constants, flags)
                    - rhs.evaluate_overflow_expr::<AB>(inputs, variables, constants, flags)
            }
            SymbolicExpr::Mul(lhs, rhs) => {
                lhs.evaluate_overflow_expr::<AB>(inputs, variables, constants, flags)
                    * rhs.evaluate_overflow_expr::<AB>(inputs, variables, constants, flags)
            }
            SymbolicExpr::Select(flag_id, lhs, rhs) => {
                let left = lhs.evaluate_overflow_expr::<AB>(inputs, variables, constants, flags);
                let right = rhs.evaluate_overflow_expr::<AB>(inputs, variables, constants, flags);
                let num_limbs = max(left.num_limbs(), right.num_limbs());
                let flag = flags[*flag_id];
                let mut res = vec![];
                for i in 0..num_limbs {
                    res.push(
                        (if i < left.num_limbs() {
                            left.limb(i).clone()
                        } else {
                            AB::Expr::ZERO
                        }) * flag.into()
                            + (if i < right.num_limbs() {
                                right.limb(i).clone()
                            } else {
                                AB::Expr::ZERO
                            }) * (AB::Expr::ONE - flag.into()),
                    );
                }
                OverflowInt::from_computed_limbs(
                    res,
                    max(left.limb_max_abs(), right.limb_max_abs()),
                    max(left.max_overflow_bits(), right.max_overflow_bits()),
                )
            }
            SymbolicExpr::Div(_, _) => unreachable!(), // Division is not allowed in constraints.
        }
    }

    /// Result will be within [0, prime).
    /// self should be a compute expression.
    /// Note that division by zero will panic.
    pub fn compute(
        &self,
        inputs: &[BigUint],
        variables: &[BigUint],
        flags: &[bool],
        prime: &BigUint,
    ) -> BigUint {
        let res = match self {
            SymbolicExpr::Input(i) => inputs[*i].clone() % prime,
            SymbolicExpr::Var(i) => variables[*i].clone(),
            SymbolicExpr::Const(_, val, _) => val.clone(),
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
                let right_inv = right.modinv(prime).unwrap();
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
        };
        assert!(
            res < prime.clone(),
            "symbolic expr: {} evaluation exceeds prime",
            self
        );
        res
    }
}
