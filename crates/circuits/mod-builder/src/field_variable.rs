use std::{
    cell::RefCell,
    cmp::{max, min},
    ops::{Add, Div, Mul, Sub},
    rc::Rc,
};

use openvm_circuit_primitives::bigint::check_carry_to_zero::get_carry_max_abs_and_bits;
use openvm_stark_backend::p3_util::log2_ceil_usize;

use super::{ExprBuilder, SymbolicExpr};

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
    pub max_carry_bits: usize,
}

impl FieldVariable {
    // Returns the index of the new variable.
    // There should be no division in the expression.
    /// This function is idempotent, i.e., if you already saved, then saving again does nothing.
    pub fn save(&mut self) -> usize {
        if let SymbolicExpr::Var(var_id) = self.expr {
            // If self.expr is already a Var, no need to save
            return var_id;
        }
        let mut builder = self.builder.borrow_mut();

        // Introduce a new variable to replace self.expr.
        let (new_var_idx, new_var) = builder.new_var();
        // self.expr - new_var = 0
        let new_constraint =
            SymbolicExpr::Sub(Box::new(self.expr.clone()), Box::new(new_var.clone()));
        // limbs information.
        builder.set_constraint(new_var_idx, new_constraint);
        builder.set_compute(new_var_idx, self.expr.clone());

        self.expr = new_var;
        self.limb_max_abs = (1 << builder.limb_bits) - 1;
        self.max_overflow_bits = builder.limb_bits;
        self.expr_limbs = builder.num_limbs;

        builder.num_variables - 1
    }

    pub fn save_output(&mut self) {
        let index = self.save();
        let mut builder = self.builder.borrow_mut();
        builder.output_indices.push(index);
    }

    pub fn canonical_limb_bits(&self) -> usize {
        self.builder.borrow().limb_bits
    }

    fn get_q_limbs(expr: SymbolicExpr, builder: &ExprBuilder) -> usize {
        let constraint_expr = SymbolicExpr::Sub(
            Box::new(expr),
            Box::new(SymbolicExpr::Var(builder.num_variables)),
        );
        let (q_limbs, _) = constraint_expr.constraint_limbs(
            &builder.prime,
            builder.limb_bits,
            builder.num_limbs,
            builder.proper_max(),
        );
        q_limbs
    }

    fn save_if_overflow(
        a: &mut FieldVariable, // will save this variable if overflow
        expr: SymbolicExpr,    /* the "compute" expression of the result variable. Note that we
                                * need to check if constraint overflows */
        limb_max_abs: usize, // The max abs of limbs of compute expression.
    ) {
        if let SymbolicExpr::Var(_) = a.expr {
            return;
        }
        let builder = a.builder.borrow();
        let canonical_limb_bits = builder.limb_bits;
        let q_limbs = FieldVariable::get_q_limbs(expr, &builder);
        let canonical_limb_max_abs = (1 << canonical_limb_bits) - 1;

        // The constraint equation is expr - new_var - qp, and we need to check if it overflows.
        let limb_max_abs = limb_max_abs
            + canonical_limb_max_abs  // new var
            + canonical_limb_max_abs * canonical_limb_max_abs * min(q_limbs, builder.num_limbs); // qp
        drop(builder);

        let max_overflow_bits = log2_ceil_usize(limb_max_abs);
        let (_, carry_bits) = get_carry_max_abs_and_bits(max_overflow_bits, canonical_limb_bits);
        if carry_bits > a.max_carry_bits {
            a.save();
        }
    }

    // TODO[Lun-Kai]: rethink about how should auto-save work.
    // This implementation requires self and other to be mutable, and might actually mutate them.
    // This might surprise the caller or introduce hard bug if the caller clone the FieldVariable
    // and then call this.
    pub fn add(&mut self, other: &mut FieldVariable) -> FieldVariable {
        assert!(Rc::ptr_eq(&self.builder, &other.builder));
        let limb_max_fn = |a: &FieldVariable, b: &FieldVariable| a.limb_max_abs + b.limb_max_abs;
        FieldVariable::save_if_overflow(
            self,
            SymbolicExpr::Add(Box::new(self.expr.clone()), Box::new(other.expr.clone())),
            limb_max_fn(self, other),
        );
        // Do again to check if the other also needs to be saved.
        FieldVariable::save_if_overflow(
            other,
            SymbolicExpr::Add(Box::new(self.expr.clone()), Box::new(other.expr.clone())),
            limb_max_fn(self, other),
        );

        let limb_max_abs = limb_max_fn(self, other);
        let max_overflow_bits = log2_ceil_usize(limb_max_abs);
        FieldVariable {
            expr: SymbolicExpr::Add(Box::new(self.expr.clone()), Box::new(other.expr.clone())),
            builder: self.builder.clone(),
            limb_max_abs,
            max_overflow_bits,
            expr_limbs: max(self.expr_limbs, other.expr_limbs),
            max_carry_bits: self.max_carry_bits,
        }
    }

    pub fn sub(&mut self, other: &mut FieldVariable) -> FieldVariable {
        assert!(Rc::ptr_eq(&self.builder, &other.builder));
        let limb_max_fn = |a: &FieldVariable, b: &FieldVariable| a.limb_max_abs + b.limb_max_abs;
        FieldVariable::save_if_overflow(
            self,
            SymbolicExpr::Sub(Box::new(self.expr.clone()), Box::new(other.expr.clone())),
            limb_max_fn(self, other),
        );
        // Do again to check if the other also needs to be saved.
        FieldVariable::save_if_overflow(
            other,
            SymbolicExpr::Sub(Box::new(self.expr.clone()), Box::new(other.expr.clone())),
            limb_max_fn(self, other),
        );

        let limb_max_abs = limb_max_fn(self, other);
        let max_overflow_bits = log2_ceil_usize(limb_max_abs);
        FieldVariable {
            expr: SymbolicExpr::Sub(Box::new(self.expr.clone()), Box::new(other.expr.clone())),
            builder: self.builder.clone(),
            limb_max_abs,
            max_overflow_bits,
            expr_limbs: max(self.expr_limbs, other.expr_limbs),
            max_carry_bits: self.max_carry_bits,
        }
    }

    pub fn mul(&mut self, other: &mut FieldVariable) -> FieldVariable {
        assert!(Rc::ptr_eq(&self.builder, &other.builder));
        let limb_max_fn = |a: &FieldVariable, b: &FieldVariable| {
            a.limb_max_abs * b.limb_max_abs * min(a.expr_limbs, b.expr_limbs)
        };
        FieldVariable::save_if_overflow(
            self,
            SymbolicExpr::Mul(Box::new(self.expr.clone()), Box::new(other.expr.clone())),
            limb_max_fn(self, other),
        );
        // Do again to check if the other also needs to be saved.
        FieldVariable::save_if_overflow(
            other,
            SymbolicExpr::Mul(Box::new(self.expr.clone()), Box::new(other.expr.clone())),
            limb_max_fn(self, other),
        );

        let limb_max_abs = limb_max_fn(self, other);
        let max_overflow_bits = log2_ceil_usize(limb_max_abs);
        FieldVariable {
            expr: SymbolicExpr::Mul(Box::new(self.expr.clone()), Box::new(other.expr.clone())),
            builder: self.builder.clone(),
            limb_max_abs,
            max_overflow_bits,
            expr_limbs: self.expr_limbs + other.expr_limbs - 1,
            max_carry_bits: self.max_carry_bits,
        }
    }

    pub fn square(&mut self) -> FieldVariable {
        let limb_max_abs = self.limb_max_abs * self.limb_max_abs * self.expr_limbs;
        FieldVariable::save_if_overflow(
            self,
            SymbolicExpr::Mul(Box::new(self.expr.clone()), Box::new(self.expr.clone())),
            limb_max_abs,
        );

        let limb_max_abs = self.limb_max_abs * self.limb_max_abs * self.expr_limbs;
        let max_overflow_bits = log2_ceil_usize(limb_max_abs);
        FieldVariable {
            expr: SymbolicExpr::Mul(Box::new(self.expr.clone()), Box::new(self.expr.clone())),
            builder: self.builder.clone(),
            limb_max_abs,
            max_overflow_bits,
            expr_limbs: self.expr_limbs * 2 - 1,
            max_carry_bits: self.max_carry_bits,
        }
    }

    pub fn int_add(&mut self, scalar: isize) -> FieldVariable {
        let limb_max_abs = self.limb_max_abs + scalar.unsigned_abs();
        FieldVariable::save_if_overflow(
            self,
            SymbolicExpr::IntAdd(Box::new(self.expr.clone()), scalar),
            limb_max_abs,
        );

        let limb_max_abs = self.limb_max_abs + scalar.unsigned_abs();
        let max_overflow_bits = log2_ceil_usize(limb_max_abs);
        FieldVariable {
            expr: SymbolicExpr::IntAdd(Box::new(self.expr.clone()), scalar),
            builder: self.builder.clone(),
            limb_max_abs,
            max_overflow_bits,
            expr_limbs: self.expr_limbs,
            max_carry_bits: self.max_carry_bits,
        }
    }

    pub fn int_mul(&mut self, scalar: isize) -> FieldVariable {
        let limb_max_abs = self.limb_max_abs * scalar.unsigned_abs();
        FieldVariable::save_if_overflow(
            self,
            SymbolicExpr::IntMul(Box::new(self.expr.clone()), scalar),
            limb_max_abs,
        );

        let limb_max_abs = self.limb_max_abs * scalar.unsigned_abs();
        let max_overflow_bits = log2_ceil_usize(limb_max_abs);
        FieldVariable {
            expr: SymbolicExpr::IntMul(Box::new(self.expr.clone()), scalar),
            builder: self.builder.clone(),
            limb_max_abs,
            max_overflow_bits,
            expr_limbs: self.expr_limbs,
            max_carry_bits: self.max_carry_bits,
        }
    }

    // expr cannot have division, so auto-save a new variable.
    // Note that division by zero will panic.
    pub fn div(&mut self, other: &mut FieldVariable) -> FieldVariable {
        assert!(Rc::ptr_eq(&self.builder, &other.builder));
        let builder = self.builder.borrow();
        let prime = builder.prime.clone();
        let limb_bits = builder.limb_bits;
        let num_limbs = builder.num_limbs;
        let proper_max = builder.proper_max().clone();
        drop(builder);

        // This is a dummy variable, will be replaced later so the index within it doesn't matter.
        // We use this to check if we need to save self/other first.
        let fake_var = SymbolicExpr::Var(0);

        // Constraint: other.expr * new_var - self.expr = 0 (mod p)
        let new_constraint = SymbolicExpr::Sub(
            Box::new(SymbolicExpr::Mul(
                Box::new(other.expr.clone()),
                Box::new(fake_var.clone()),
            )),
            Box::new(self.expr.clone()),
        );
        let carry_bits =
            new_constraint.constraint_carry_bits_with_pq(&prime, limb_bits, num_limbs, &proper_max);
        if carry_bits > self.max_carry_bits {
            self.save();
        }
        // Do it again to check if other needs to be saved.
        let new_constraint = SymbolicExpr::Sub(
            Box::new(SymbolicExpr::Mul(
                Box::new(other.expr.clone()),
                Box::new(fake_var.clone()),
            )),
            Box::new(self.expr.clone()),
        );
        let carry_bits =
            new_constraint.constraint_carry_bits_with_pq(&prime, limb_bits, num_limbs, &proper_max);
        if carry_bits > self.max_carry_bits {
            other.save();
        }

        let mut builder = self.builder.borrow_mut();
        let (new_var_idx, new_var) = builder.new_var();
        let new_constraint = SymbolicExpr::Sub(
            Box::new(SymbolicExpr::Mul(
                Box::new(other.expr.clone()),
                Box::new(new_var.clone()),
            )),
            Box::new(self.expr.clone()),
        );
        builder.set_constraint(new_var_idx, new_constraint);
        // Only compute can have division.
        let compute = SymbolicExpr::Div(Box::new(self.expr.clone()), Box::new(other.expr.clone()));
        builder.set_compute(new_var_idx, compute);
        drop(builder);

        FieldVariable::from_var(self.builder.clone(), new_var)
    }

    pub fn from_var(builder: Rc<RefCell<ExprBuilder>>, var: SymbolicExpr) -> FieldVariable {
        let borrowed_builder = builder.borrow();
        let max_carry_bits = borrowed_builder.max_carry_bits;
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
            max_carry_bits,
        }
    }

    pub fn select(flag_id: usize, a: &FieldVariable, b: &FieldVariable) -> FieldVariable {
        assert!(Rc::ptr_eq(&a.builder, &b.builder));
        let limb_max_abs = max(a.limb_max_abs, b.limb_max_abs);
        let max_overflow_bits = max(a.max_overflow_bits, b.max_overflow_bits);
        let expr_limbs = max(a.expr_limbs, b.expr_limbs);
        FieldVariable {
            expr: SymbolicExpr::Select(flag_id, Box::new(a.expr.clone()), Box::new(b.expr.clone())),
            builder: a.builder.clone(),
            limb_max_abs,
            max_overflow_bits,
            expr_limbs,
            max_carry_bits: a.max_carry_bits,
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

// Note that division by zero will panic.
impl Div<FieldVariable> for FieldVariable {
    type Output = FieldVariable;

    fn div(mut self, mut rhs: FieldVariable) -> Self::Output {
        let x = &mut self;
        x.div(&mut rhs)
    }
}

impl Div<&mut FieldVariable> for &mut FieldVariable {
    type Output = FieldVariable;

    fn div(self, rhs: &mut FieldVariable) -> Self::Output {
        FieldVariable::div(self, rhs)
    }
}
