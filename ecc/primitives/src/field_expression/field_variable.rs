use std::{
    cell::RefCell,
    ops::{Add, Div, Mul, Sub},
    rc::Rc,
};

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
}

impl FieldVariable {
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
    }

    // no carry (no automaticly save when limbs overflow)
    pub fn add(&self, other: &FieldVariable) -> FieldVariable {
        assert!(Rc::ptr_eq(&self.builder, &other.builder));
        FieldVariable {
            expr: SymbolicExpr::Add(Box::new(self.expr.clone()), Box::new(other.expr.clone())),
            builder: self.builder.clone(),
        }
    }

    // no carry (no automaticly save when limbs overflow)
    pub fn sub(&self, other: &FieldVariable) -> FieldVariable {
        assert!(Rc::ptr_eq(&self.builder, &other.builder));
        FieldVariable {
            expr: SymbolicExpr::Sub(Box::new(self.expr.clone()), Box::new(other.expr.clone())),
            builder: self.builder.clone(),
        }
    }

    // no carry (no automaticly save when limbs overflow)
    pub fn mul(&self, other: &FieldVariable) -> FieldVariable {
        assert!(Rc::ptr_eq(&self.builder, &other.builder));
        FieldVariable {
            expr: SymbolicExpr::Mul(Box::new(self.expr.clone()), Box::new(other.expr.clone())),
            builder: self.builder.clone(),
        }
    }

    // TODO: should check that scalar is within the range of limb bits. But we don't have limb bits here, might need to do it in eval/compute.
    // no carry (no automaticly save when limbs overflow)
    pub fn int_mul(&self, scalar: isize) -> FieldVariable {
        FieldVariable {
            expr: SymbolicExpr::IntMul(Box::new(self.expr.clone()), scalar),
            builder: self.builder.clone(),
        }
    }

    // expr cannot have division, so auto-save a new variable.
    pub fn div(&self, other: &FieldVariable) -> FieldVariable {
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
        }
    }
}

// TODO: these operations should auto-carry.
impl Add for FieldVariable {
    type Output = FieldVariable;

    fn add(self, rhs: FieldVariable) -> Self::Output {
        self.add(&rhs)
    }
}

impl Add<FieldVariable> for &FieldVariable {
    type Output = FieldVariable;

    fn add(self, rhs: FieldVariable) -> Self::Output {
        self.add(&rhs)
    }
}

impl Add<&FieldVariable> for FieldVariable {
    type Output = FieldVariable;

    fn add(self, rhs: &FieldVariable) -> Self::Output {
        FieldVariable::add(&self, rhs)
    }
}

impl Sub for FieldVariable {
    type Output = FieldVariable;

    fn sub(self, rhs: FieldVariable) -> Self::Output {
        self.sub(&rhs)
    }
}

impl Sub<FieldVariable> for &FieldVariable {
    type Output = FieldVariable;

    fn sub(self, rhs: FieldVariable) -> Self::Output {
        self.sub(&rhs)
    }
}

impl Sub<&FieldVariable> for FieldVariable {
    type Output = FieldVariable;

    fn sub(self, rhs: &FieldVariable) -> Self::Output {
        FieldVariable::sub(&self, rhs)
    }
}

impl Mul for FieldVariable {
    type Output = FieldVariable;

    fn mul(self, rhs: FieldVariable) -> Self::Output {
        self.mul(&rhs)
    }
}

impl Mul<FieldVariable> for &FieldVariable {
    type Output = FieldVariable;

    fn mul(self, rhs: FieldVariable) -> Self::Output {
        self.mul(&rhs)
    }
}

impl Mul<&FieldVariable> for FieldVariable {
    type Output = FieldVariable;

    fn mul(self, rhs: &FieldVariable) -> Self::Output {
        FieldVariable::mul(&self, rhs)
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
