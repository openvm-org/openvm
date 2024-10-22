use std::{
    cell::RefCell,
    cmp::{max, min},
    ops::{Add, Div, Mul, Sub},
    rc::Rc,
};

use afs_primitives::bigint::check_carry_to_zero::get_carry_max_abs_and_bits;
use p3_util::log2_ceil_usize;

use super::{ExprBuilder, SymbolicExpr};

pub trait FieldVariableConfig {
    // This is the limb bits for a canonical field element. Typically 8.
    fn canonical_limb_bits() -> usize;
    // The max bits allowed per limb, determined by the underlying field we use to represent the field element.
    // For example BabyBear -> 29.
    fn max_limb_bits() -> usize;
    // Number of limbs to represent a field element.
    fn num_limbs_per_field_element() -> usize;
}

#[derive(Clone)]
pub struct FieldVariable {
    // 1. This will be "reset" to Var(n), when calling save on it.
    // 2. This is an expression to "compute" (instead of to "constrain")
    // But it will NOT have division, as it will be auto save and reset.
    // For example, if we want to compute d = a * b + c, the expr here will be a * b + c
    // So this is not a constraint that should be equal to zero (a * b + c - d is the constraint).
    pub expr: SymbolicExpr,

    pub builder: Rc<RefCell<ExprBuilder>>,

    // Limb related information when evaluated as an OverflowInt (vector of limbs).
    // Max abs of each limb.
    pub limb_max_abs: usize,
    // All limbs should be within [-2^max_overflow_bits, 2^max_overflow_bits)
    // This is log2_ceil(limb_max_abs)
    pub max_overflow_bits: usize,
    // Number of limbs to represent the expression.
    pub expr_limbs: usize,

    // This is the same for all FieldVariable, but we might use different values at runtime,
    // so store it here for easy configuration.
    pub range_checker_bits: usize,
}

impl FieldVariable {
    // Returns the index of the new variable.
    // There should be no division in the expression.
    pub fn save(&mut self) -> usize {
        if let SymbolicExpr::Var(_) = self.expr {
            // If self.expr is already a Var, no need to save
            return self.builder.borrow().num_variables - 1;
        }
        let mut builder = self.builder.borrow_mut();
        builder.num_variables += 1;

        // Introduce a new variable to replace self.expr.
        let new_var = SymbolicExpr::Var(builder.num_variables - 1);
        // self.expr - new_var = 0
        let new_constraint =
            SymbolicExpr::Sub(Box::new(self.expr.clone()), Box::new(new_var.clone()));
        // limbs information.
        let (q_limbs, carry_limbs) =
            self.expr
                .constraint_limbs(&builder.prime, builder.limb_bits, builder.num_limbs);
        builder.constraints.push(new_constraint);
        builder.q_limbs.push(q_limbs);
        builder.carry_limbs.push(carry_limbs);
        builder.computes.push(self.expr.clone());

        self.expr = new_var;
        self.limb_max_abs = (1 << builder.limb_bits) - 1;
        self.max_overflow_bits = builder.limb_bits;
        self.expr_limbs = builder.num_limbs;

        builder.num_variables - 1
    }

    pub fn canonical_limb_bits(&self) -> usize {
        self.builder.borrow().limb_bits
    }

    fn save_if_overflow(
        a: &mut FieldVariable,
        b: &mut FieldVariable,
        limb_max_fn: fn(&FieldVariable, &FieldVariable) -> usize,
    ) {
        let canonical_limb_bits = a.builder.borrow().limb_bits;
        debug_assert_eq!(canonical_limb_bits, b.builder.borrow().limb_bits);
        let limb_max_abs = limb_max_fn(a, b);
        let max_overflow_bits = log2_ceil_usize(limb_max_abs);
        let (_, carry_bits) = get_carry_max_abs_and_bits(max_overflow_bits, canonical_limb_bits);
        if carry_bits > a.range_checker_bits {
            // Need to save self or other (or both) to prevent overflow.
            if a.max_overflow_bits > b.max_overflow_bits {
                assert!(a.max_overflow_bits > canonical_limb_bits);
                a.save();
            } else {
                assert!(b.max_overflow_bits > canonical_limb_bits);
                b.save();
            }
        }
    }

    // TODO: rethink about how should auto-save work.
    // This implementation requires self and other to be mutable, and might actually mutate them.
    // This might surprise the caller or introduce hard bug if the caller clone the FieldVariable and then call this.
    pub fn add(&mut self, other: &mut FieldVariable) -> FieldVariable {
        assert!(Rc::ptr_eq(&self.builder, &other.builder));
        let limb_max_fn = |a: &FieldVariable, b: &FieldVariable| a.limb_max_abs + b.limb_max_abs;
        FieldVariable::save_if_overflow(self, other, limb_max_fn);
        // Do again to check if the other also needs to be saved.
        FieldVariable::save_if_overflow(self, other, limb_max_fn);

        let limb_max_abs = limb_max_fn(self, other);
        let max_overflow_bits = log2_ceil_usize(limb_max_abs);
        FieldVariable {
            expr: SymbolicExpr::Add(Box::new(self.expr.clone()), Box::new(other.expr.clone())),
            builder: self.builder.clone(),
            limb_max_abs,
            max_overflow_bits,
            expr_limbs: max(self.expr_limbs, other.expr_limbs),
            range_checker_bits: self.range_checker_bits,
        }
    }

    pub fn sub(&mut self, other: &mut FieldVariable) -> FieldVariable {
        assert!(Rc::ptr_eq(&self.builder, &other.builder));
        let limb_max_fn = |a: &FieldVariable, b: &FieldVariable| a.limb_max_abs + b.limb_max_abs;
        FieldVariable::save_if_overflow(self, other, limb_max_fn);
        // Do again to check if the other also needs to be saved.
        FieldVariable::save_if_overflow(self, other, limb_max_fn);

        let limb_max_abs = limb_max_fn(self, other);
        let max_overflow_bits = log2_ceil_usize(limb_max_abs);
        FieldVariable {
            expr: SymbolicExpr::Sub(Box::new(self.expr.clone()), Box::new(other.expr.clone())),
            builder: self.builder.clone(),
            limb_max_abs,
            max_overflow_bits,
            expr_limbs: max(self.expr_limbs, other.expr_limbs),
            range_checker_bits: self.range_checker_bits,
        }
    }

    pub fn mul(&mut self, other: &mut FieldVariable) -> FieldVariable {
        assert!(Rc::ptr_eq(&self.builder, &other.builder));
        let limb_max_fn = |a: &FieldVariable, b: &FieldVariable| {
            a.limb_max_abs * b.limb_max_abs * min(a.expr_limbs, b.expr_limbs)
        };
        FieldVariable::save_if_overflow(self, other, limb_max_fn);
        // Do again to check if the other also needs to be saved.
        FieldVariable::save_if_overflow(self, other, limb_max_fn);

        let limb_max_abs = limb_max_fn(self, other);
        let max_overflow_bits = log2_ceil_usize(limb_max_abs);
        FieldVariable {
            expr: SymbolicExpr::Mul(Box::new(self.expr.clone()), Box::new(other.expr.clone())),
            builder: self.builder.clone(),
            limb_max_abs,
            max_overflow_bits,
            expr_limbs: self.expr_limbs + other.expr_limbs - 1,
            range_checker_bits: self.range_checker_bits,
        }
    }

    pub fn square(&mut self) -> FieldVariable {
        let builder = self.builder.borrow();
        let limb_max_abs = self.limb_max_abs * self.limb_max_abs * self.expr_limbs;
        let max_overflow_bits = log2_ceil_usize(limb_max_abs);
        let (_, carry_bits) = get_carry_max_abs_and_bits(max_overflow_bits, builder.limb_bits);
        drop(builder);
        if carry_bits > self.range_checker_bits {
            self.save();
        }

        let limb_max_abs = self.limb_max_abs * self.limb_max_abs * self.expr_limbs;
        let max_overflow_bits = log2_ceil_usize(limb_max_abs);
        FieldVariable {
            expr: SymbolicExpr::Mul(Box::new(self.expr.clone()), Box::new(self.expr.clone())),
            builder: self.builder.clone(),
            limb_max_abs,
            max_overflow_bits,
            expr_limbs: self.expr_limbs * 2 - 1,
            range_checker_bits: self.range_checker_bits,
        }
    }

    pub fn int_mul(&mut self, scalar: isize) -> FieldVariable {
        let builder = self.builder.borrow();
        let max_limb_bits = builder.max_limb_bits;
        assert!(scalar.unsigned_abs() < (1 << max_limb_bits));
        let limb_max_abs = self.limb_max_abs * scalar.unsigned_abs();
        let max_overflow_bits = log2_ceil_usize(limb_max_abs);
        let (_, carry_bits) = get_carry_max_abs_and_bits(max_overflow_bits, builder.limb_bits);
        drop(builder);
        if carry_bits > self.range_checker_bits {
            self.save();
        }
        let limb_max_abs = self.limb_max_abs * scalar.unsigned_abs();
        let max_overflow_bits = log2_ceil_usize(limb_max_abs);
        let mut res = FieldVariable {
            expr: SymbolicExpr::IntMul(Box::new(self.expr.clone()), scalar),
            builder: self.builder.clone(),
            limb_max_abs,
            max_overflow_bits,
            expr_limbs: self.expr_limbs,
            range_checker_bits: self.range_checker_bits,
        };
        if max_overflow_bits > max_limb_bits {
            res.save();
        }
        res
    }

    // expr cannot have division, so auto-save a new variable.
    pub fn div(&self, other: &FieldVariable) -> FieldVariable {
        assert!(Rc::ptr_eq(&self.builder, &other.builder));
        let new_var = {
            let mut builder = self.builder.borrow_mut();
            // Introduce a new variable to replace self.expr / other.expr.
            let new_var = builder.new_var();
            // other.expr * new_var = self.expr
            let new_constraint = SymbolicExpr::Sub(
                Box::new(SymbolicExpr::Mul(
                    Box::new(other.expr.clone()),
                    Box::new(new_var.clone()),
                )),
                Box::new(self.expr.clone()),
            );
            builder.add_constraint(new_constraint);
            // Only compute can have division.
            let compute =
                SymbolicExpr::Div(Box::new(self.expr.clone()), Box::new(other.expr.clone()));
            builder.computes.push(compute);
            new_var
        };

        FieldVariable::from_var(self.builder.clone(), new_var)
    }

    pub fn from_var(builder: Rc<RefCell<ExprBuilder>>, var: SymbolicExpr) -> FieldVariable {
        let borrowed_builder = builder.borrow();
        let range_checker_bits = borrowed_builder.range_checker_bits;
        assert!(
            matches!(var, SymbolicExpr::Var(_)),
            "Expected var to be of type SymbolicExpr::Var"
        );
        let num_limbs = borrowed_builder.num_limbs;
        let canonical_limb_bits = borrowed_builder.limb_bits;
        drop(borrowed_builder);
        FieldVariable {
            expr: var,
            builder,
            limb_max_abs: (1 << canonical_limb_bits) - 1,
            max_overflow_bits: canonical_limb_bits,
            expr_limbs: num_limbs,
            range_checker_bits,
        }
    }

    pub fn select(flag_id: usize, a: &FieldVariable, b: &FieldVariable) -> FieldVariable {
        assert!(Rc::ptr_eq(&a.builder, &b.builder));
        let left_limb_max_abs = max(a.limb_max_abs, b.limb_max_abs);
        let left_max_overflow_bits = max(a.max_overflow_bits, b.max_overflow_bits);
        let left_expr_limbs = max(a.expr_limbs, b.expr_limbs);
        let right_limb_max_abs = left_limb_max_abs;
        let right_max_overflow_bits = left_max_overflow_bits;
        let right_expr_limbs = left_expr_limbs;
        assert_eq!(left_limb_max_abs, right_limb_max_abs);
        assert_eq!(left_max_overflow_bits, right_max_overflow_bits);
        assert_eq!(left_expr_limbs, right_expr_limbs);
        FieldVariable {
            expr: SymbolicExpr::Select(flag_id, Box::new(a.expr.clone()), Box::new(b.expr.clone())),
            builder: a.builder.clone(),
            limb_max_abs: left_limb_max_abs,
            max_overflow_bits: left_max_overflow_bits,
            expr_limbs: left_expr_limbs,
            range_checker_bits: a.range_checker_bits,
        }
    }
}

impl Add<&mut FieldVariable> for &mut FieldVariable {
    type Output = FieldVariable;

    fn add(self, rhs: &mut FieldVariable) -> Self::Output {
        self.add(rhs)
    }
}

impl Add<FieldVariable> for FieldVariable {
    type Output = FieldVariable;

    fn add(mut self, mut rhs: FieldVariable) -> Self::Output {
        let x = &mut self;
        x.add(&mut rhs)
    }
}

impl Sub<FieldVariable> for FieldVariable {
    type Output = FieldVariable;

    fn sub(mut self, mut rhs: FieldVariable) -> Self::Output {
        let x = &mut self;
        x.sub(&mut rhs)
    }
}

impl Sub<&mut FieldVariable> for &mut FieldVariable {
    type Output = FieldVariable;

    fn sub(self, rhs: &mut FieldVariable) -> Self::Output {
        self.sub(rhs)
    }
}

impl Mul<FieldVariable> for FieldVariable {
    type Output = FieldVariable;

    fn mul(mut self, mut rhs: FieldVariable) -> Self::Output {
        let x = &mut self;
        x.mul(&mut rhs)
    }
}

impl Mul<&mut FieldVariable> for &mut FieldVariable {
    type Output = FieldVariable;

    fn mul(self, rhs: &mut FieldVariable) -> Self::Output {
        FieldVariable::mul(self, rhs)
    }
}

impl Div for FieldVariable {
    type Output = FieldVariable;

    fn div(self, rhs: FieldVariable) -> Self::Output {
        self.div(&rhs)
    }
}

impl Div<FieldVariable> for &FieldVariable {
    type Output = FieldVariable;

    fn div(self, rhs: FieldVariable) -> Self::Output {
        self.div(&rhs)
    }
}

impl Div<&FieldVariable> for FieldVariable {
    type Output = FieldVariable;

    fn div(self, rhs: &FieldVariable) -> Self::Output {
        FieldVariable::div(&self, rhs)
    }
}

impl Div<&FieldVariable> for &FieldVariable {
    type Output = FieldVariable;

    fn div(self, rhs: &FieldVariable) -> Self::Output {
        FieldVariable::div(self, rhs)
    }
}
