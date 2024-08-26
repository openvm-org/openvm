use std::{
    marker::PhantomData,
    ops::{AddAssign, MulAssign},
};

use p3_field::{AbstractField, ExtensionField, Field};
use p3_matrix::Matrix;
use p3_uni_stark::{StarkGenericConfig, Val};

use super::{
    symbolic::{
        symbolic_expression::{SymbolicEvaluator, SymbolicExpression},
        symbolic_variable::{Entry, SymbolicVariable},
    },
    ViewPair,
};

pub type VerifierConstraintFolder<'a, SC> = GenericVerifierConstraintFolder<
    'a,
    Val<SC>,
    <SC as StarkGenericConfig>::Challenge,
    Val<SC>,
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenge,
>;
// Struct definition copied from sp1 under MIT license.
/// A folder for verifier constraints with generic types.
///
/// `Var` is still a challenge type because this is a verifier.
pub struct GenericVerifierConstraintFolder<'a, F, EF, PubVar, Var, Expr> {
    pub preprocessed: ViewPair<'a, Var>,
    pub partitioned_main: Vec<ViewPair<'a, Var>>,
    pub after_challenge: Vec<ViewPair<'a, Var>>,
    pub challenges: &'a [Vec<Var>],
    pub is_first_row: Var,
    pub is_last_row: Var,
    pub is_transition: Var,
    pub alpha: Var,
    pub accumulator: Expr,
    pub public_values: &'a [PubVar],
    pub exposed_values_after_challenge: &'a [Vec<Var>],
    pub _marker: PhantomData<(F, EF)>,
}

impl<'a, F, EF, PubVar, Var, Expr> GenericVerifierConstraintFolder<'a, F, EF, PubVar, Var, Expr>
where
    F: Field,
    EF: ExtensionField<F>,
    Expr: AbstractField + From<F> + MulAssign<Var> + AddAssign<Var>,
    Var: Into<Expr> + Copy + Send + Sync,
    PubVar: Into<Expr> + Copy,
{
    pub fn eval_constraints(&mut self, constraints: &[SymbolicExpression<F>]) {
        for constraint in constraints {
            let x = self.eval_expr(constraint);
            self.assert_zero(x);
        }
    }

    pub fn assert_zero(&mut self, x: impl Into<Expr>) {
        let x = x.into();
        self.accumulator *= self.alpha;
        self.accumulator += x;
    }
}

impl<'a, F, EF, PubVar, Var, Expr> SymbolicEvaluator<F, Expr>
    for GenericVerifierConstraintFolder<'a, F, EF, PubVar, Var, Expr>
where
    F: Field,
    EF: ExtensionField<F>,
    Expr: AbstractField + From<F>,
    Var: Into<Expr> + Copy + Send + Sync,
    PubVar: Into<Expr> + Copy,
{
    fn eval_var(&self, symbolic_var: SymbolicVariable<F>) -> Expr {
        let index = symbolic_var.index;
        match symbolic_var.entry {
            Entry::Preprocessed { offset } => self.preprocessed.get(offset, index).into(),
            Entry::Main { part_index, offset } => {
                self.partitioned_main[part_index].get(offset, index).into()
            }
            Entry::Public => self.public_values[index].into(),
            Entry::Permutation { offset } => self
                .after_challenge
                .first()
                .expect("Challenge phase not supported")
                .get(offset, index)
                .into(),
            Entry::Challenge => self
                .challenges
                .first()
                .expect("Challenge phase not supported")[index]
                .into(),
            Entry::Exposed => self
                .exposed_values_after_challenge
                .first()
                .expect("Challenge phase not supported")[index]
                .into(),
        }
    }

    fn eval_expr(&self, symbolic_expr: &SymbolicExpression<F>) -> Expr {
        // TODO[jpw] don't use recursion to avoid stack overflow
        match symbolic_expr {
            SymbolicExpression::Variable(var) => self.eval_var(*var),
            SymbolicExpression::Constant(c) => (*c).into(),
            SymbolicExpression::Add { x, y, .. } => self.eval_expr(x) + self.eval_expr(y),
            SymbolicExpression::Sub { x, y, .. } => self.eval_expr(x) - self.eval_expr(y),
            SymbolicExpression::Neg { x, .. } => -self.eval_expr(x),
            SymbolicExpression::Mul { x, y, .. } => self.eval_expr(x) * self.eval_expr(y),
            SymbolicExpression::IsFirstRow => self.is_first_row.into(),
            SymbolicExpression::IsLastRow => self.is_last_row.into(),
            SymbolicExpression::IsTransition => self.is_transition.into(),
        }
    }
}
