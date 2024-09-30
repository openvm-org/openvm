use std::{
    cell::RefCell,
    cmp::{max, min},
    marker::PhantomData,
    ops::{Add, Div, Mul, Sub},
    rc::Rc,
};

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
pub struct FieldVariable<C: FieldVariableConfig> {
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

    pub _marker: PhantomData<C>,
}

impl<C: FieldVariableConfig> FieldVariable<C> {
    // There should be no division in the expression.
    pub fn save(&mut self) {
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
        self.limb_max_abs = (1 << C::canonical_limb_bits()) - 1;
        self.max_overflow_bits = C::canonical_limb_bits();
        self.expr_limbs = C::num_limbs_per_field_element();
    }

    pub fn add(&self, other: &FieldVariable<C>) -> FieldVariable<C> {
        assert!(Rc::ptr_eq(&self.builder, &other.builder));
        let limb_max_abs = self.limb_max_abs + other.limb_max_abs;
        let max_overflow_bits = log2_ceil_usize(limb_max_abs);
        let mut res = FieldVariable {
            expr: SymbolicExpr::Add(Box::new(self.expr.clone()), Box::new(other.expr.clone())),
            builder: self.builder.clone(),
            limb_max_abs,
            max_overflow_bits,
            expr_limbs: max(self.expr_limbs, other.expr_limbs),
            _marker: PhantomData,
        };
        if max_overflow_bits > C::max_limb_bits() {
            res.save();
        }
        res
    }

    pub fn sub(&self, other: &FieldVariable<C>) -> FieldVariable<C> {
        assert!(Rc::ptr_eq(&self.builder, &other.builder));
        let limb_max_abs = self.limb_max_abs + other.limb_max_abs;
        let max_overflow_bits = log2_ceil_usize(limb_max_abs);
        let mut res = FieldVariable {
            expr: SymbolicExpr::Sub(Box::new(self.expr.clone()), Box::new(other.expr.clone())),
            builder: self.builder.clone(),
            limb_max_abs,
            max_overflow_bits,
            expr_limbs: max(self.expr_limbs, other.expr_limbs),
            _marker: PhantomData,
        };
        if max_overflow_bits > C::max_limb_bits() {
            res.save();
        }
        res
    }

    pub fn mul(&self, other: &FieldVariable<C>) -> FieldVariable<C> {
        assert!(Rc::ptr_eq(&self.builder, &other.builder));
        let limb_max_abs =
            self.limb_max_abs * other.limb_max_abs * min(self.expr_limbs, other.expr_limbs);
        let max_overflow_bits = log2_ceil_usize(limb_max_abs);
        let mut res = FieldVariable {
            expr: SymbolicExpr::Mul(Box::new(self.expr.clone()), Box::new(other.expr.clone())),
            builder: self.builder.clone(),
            limb_max_abs,
            max_overflow_bits,
            expr_limbs: self.expr_limbs + other.expr_limbs - 1,
            _marker: PhantomData,
        };
        if max_overflow_bits > C::max_limb_bits() {
            res.save();
        }
        res
    }

    pub fn int_mul(&self, scalar: isize) -> FieldVariable<C> {
        assert!(scalar.unsigned_abs() < (1 << C::max_limb_bits()));
        let limb_max_abs = self.limb_max_abs * scalar.unsigned_abs();
        let max_overflow_bits = log2_ceil_usize(limb_max_abs);
        let mut res = FieldVariable {
            expr: SymbolicExpr::IntMul(Box::new(self.expr.clone()), scalar),
            builder: self.builder.clone(),
            limb_max_abs,
            max_overflow_bits,
            expr_limbs: self.expr_limbs,
            _marker: PhantomData,
        };
        if max_overflow_bits > C::max_limb_bits() {
            res.save();
        }
        res
    }

    // expr cannot have division, so auto-save a new variable.
    pub fn div(&self, other: &FieldVariable<C>) -> FieldVariable<C> {
        assert!(Rc::ptr_eq(&self.builder, &other.builder));
        let mut builder = self.builder.borrow_mut();
        builder.num_variables += 1;

        // Introduce a new variable to replace self.expr / other.expr.
        let new_var = SymbolicExpr::Var(builder.num_variables - 1);
        // other.expr * new_var = self.expr
        let new_constraint = SymbolicExpr::Sub(
            Box::new(SymbolicExpr::Mul(
                Box::new(other.expr.clone()),
                Box::new(new_var.clone()),
            )),
            Box::new(self.expr.clone()),
        );
        // limbs information.
        let (q_limbs, carry_limbs) =
            new_constraint.constraint_limbs(&builder.prime, builder.limb_bits, builder.num_limbs);
        builder.constraints.push(new_constraint);
        builder.q_limbs.push(q_limbs);
        builder.carry_limbs.push(carry_limbs);

        // Only compute can have division.
        let compute = SymbolicExpr::Div(Box::new(self.expr.clone()), Box::new(other.expr.clone()));
        builder.computes.push(compute);

        FieldVariable {
            expr: new_var,
            builder: self.builder.clone(),
            limb_max_abs: (1 << C::canonical_limb_bits()) - 1,
            max_overflow_bits: C::canonical_limb_bits(),
            expr_limbs: C::num_limbs_per_field_element(),
            _marker: PhantomData,
        }
    }
}

impl<C: FieldVariableConfig> Add for FieldVariable<C> {
    type Output = FieldVariable<C>;

    fn add(self, rhs: FieldVariable<C>) -> Self::Output {
        self.add(&rhs)
    }
}

impl<C: FieldVariableConfig> Add<FieldVariable<C>> for &FieldVariable<C> {
    type Output = FieldVariable<C>;

    fn add(self, rhs: FieldVariable<C>) -> Self::Output {
        self.add(&rhs)
    }
}

impl<C: FieldVariableConfig> Add<&FieldVariable<C>> for FieldVariable<C> {
    type Output = FieldVariable<C>;

    fn add(self, rhs: &FieldVariable<C>) -> Self::Output {
        FieldVariable::add(&self, rhs)
    }
}

impl<C: FieldVariableConfig> Sub<FieldVariable<C>> for FieldVariable<C> {
    type Output = FieldVariable<C>;

    fn sub(self, rhs: FieldVariable<C>) -> Self::Output {
        self.sub(&rhs)
    }
}

impl<C: FieldVariableConfig> Sub<FieldVariable<C>> for &FieldVariable<C> {
    type Output = FieldVariable<C>;

    fn sub(self, rhs: FieldVariable<C>) -> Self::Output {
        self.sub(&rhs)
    }
}

impl<C: FieldVariableConfig> Sub<&FieldVariable<C>> for FieldVariable<C> {
    type Output = FieldVariable<C>;

    fn sub(self, rhs: &FieldVariable<C>) -> Self::Output {
        FieldVariable::sub(&self, rhs)
    }
}

impl<C: FieldVariableConfig> Mul for FieldVariable<C> {
    type Output = FieldVariable<C>;

    fn mul(self, rhs: FieldVariable<C>) -> Self::Output {
        self.mul(&rhs)
    }
}

impl<C: FieldVariableConfig> Mul<FieldVariable<C>> for &FieldVariable<C> {
    type Output = FieldVariable<C>;

    fn mul(self, rhs: FieldVariable<C>) -> Self::Output {
        self.mul(&rhs)
    }
}

impl<C: FieldVariableConfig> Mul<&FieldVariable<C>> for FieldVariable<C> {
    type Output = FieldVariable<C>;

    fn mul(self, rhs: &FieldVariable<C>) -> Self::Output {
        FieldVariable::mul(&self, rhs)
    }
}

impl<C: FieldVariableConfig> Div for FieldVariable<C> {
    type Output = FieldVariable<C>;

    fn div(self, rhs: FieldVariable<C>) -> Self::Output {
        self.div(&rhs)
    }
}

impl<C: FieldVariableConfig> Div<FieldVariable<C>> for &FieldVariable<C> {
    type Output = FieldVariable<C>;

    fn div(self, rhs: FieldVariable<C>) -> Self::Output {
        self.div(&rhs)
    }
}

impl<C: FieldVariableConfig> Div<&FieldVariable<C>> for FieldVariable<C> {
    type Output = FieldVariable<C>;

    fn div(self, rhs: &FieldVariable<C>) -> Self::Output {
        FieldVariable::div(&self, rhs)
    }
}
