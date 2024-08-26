use p3_air::{
    AirBuilder, AirBuilderWithPublicValues, ExtensionBuilder, PairBuilder, PermutationAirBuilder,
};
use p3_field::AbstractField;
use p3_matrix::Matrix;
use p3_uni_stark::{StarkGenericConfig, Val};

use super::{
    symbolic::{
        symbolic_expression::{SymbolicEvaluator, SymbolicExpression},
        symbolic_variable::{Entry, SymbolicVariable},
    },
    ViewPair,
};
use crate::rap::PermutationAirBuilderWithExposedValues;

pub struct VerifierConstraintFolder<'a, SC: StarkGenericConfig> {
    pub preprocessed: ViewPair<'a, SC::Challenge>,
    pub partitioned_main: Vec<ViewPair<'a, SC::Challenge>>,
    pub after_challenge: Vec<ViewPair<'a, SC::Challenge>>,
    pub challenges: &'a [Vec<SC::Challenge>],
    pub is_first_row: SC::Challenge,
    pub is_last_row: SC::Challenge,
    pub is_transition: SC::Challenge,
    pub alpha: SC::Challenge,
    pub accumulator: SC::Challenge,
    pub public_values: &'a [Val<SC>],
    pub exposed_values_after_challenge: &'a [Vec<SC::Challenge>],
}

impl<'a, SC: StarkGenericConfig> VerifierConstraintFolder<'a, SC> {
    pub fn eval_constraints(&mut self, constraints: &[SymbolicExpression<Val<SC>>]) {
        for constraint in constraints {
            let x = self.eval_expr(constraint);
            self.assert_zero(x);
        }
    }
}

impl<'a, SC> SymbolicEvaluator<Val<SC>, SC::Challenge> for VerifierConstraintFolder<'a, SC>
where
    SC: StarkGenericConfig,
{
    fn eval_var(&self, symbolic_var: SymbolicVariable<Val<SC>>) -> SC::Challenge {
        let index = symbolic_var.index;
        match symbolic_var.entry {
            Entry::Preprocessed { offset } => self.preprocessed.get(offset, index),
            Entry::Main { part_index, offset } => {
                self.partitioned_main[part_index].get(offset, index)
            }
            Entry::Public => self.public_values[index].into(),
            Entry::Permutation { offset } => self.permutation().get(offset, index),
            Entry::Challenge => self.permutation_randomness()[index],
            Entry::Exposed => self.permutation_exposed_values()[index],
        }
    }

    fn eval_expr(&self, symbolic_expr: &SymbolicExpression<Val<SC>>) -> SC::Challenge {
        // TODO[jpw] don't use recursion to avoid stack overflow
        match symbolic_expr {
            SymbolicExpression::Variable(var) => self.eval_var(*var),
            SymbolicExpression::Constant(c) => (*c).into(),
            SymbolicExpression::Add { x, y, .. } => self.eval_expr(x) + self.eval_expr(y),
            SymbolicExpression::Sub { x, y, .. } => self.eval_expr(x) - self.eval_expr(y),
            SymbolicExpression::Neg { x, .. } => -self.eval_expr(x),
            SymbolicExpression::Mul { x, y, .. } => self.eval_expr(x) * self.eval_expr(y),
            SymbolicExpression::IsFirstRow => self.is_first_row,
            SymbolicExpression::IsLastRow => self.is_last_row,
            SymbolicExpression::IsTransition => self.is_transition,
        }
    }
}

// AirBuilder is no longer needed, but we keep it for documentation purposes.
impl<'a, SC: StarkGenericConfig> AirBuilder for VerifierConstraintFolder<'a, SC> {
    type F = Val<SC>;
    type Expr = SC::Challenge;
    type Var = SC::Challenge;
    type M = ViewPair<'a, SC::Challenge>;

    /// It is difficult to horizontally concatenate matrices when the main trace is partitioned, so we disable this method in that case.
    fn main(&self) -> Self::M {
        if self.partitioned_main.len() == 1 {
            self.partitioned_main[0]
        } else {
            panic!("Main trace is either empty or partitioned. This function should not be used.")
        }
    }

    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.is_transition
        } else {
            panic!("uni-stark only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let x: SC::Challenge = x.into();
        self.accumulator *= self.alpha;
        self.accumulator += x;
    }
}

impl<'a, SC> PairBuilder for VerifierConstraintFolder<'a, SC>
where
    SC: StarkGenericConfig,
{
    fn preprocessed(&self) -> Self::M {
        self.preprocessed
    }
}

impl<'a, SC> ExtensionBuilder for VerifierConstraintFolder<'a, SC>
where
    SC: StarkGenericConfig,
{
    type EF = SC::Challenge;
    type ExprEF = SC::Challenge;
    type VarEF = SC::Challenge;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        let x: SC::Challenge = x.into();
        self.accumulator *= SC::Challenge::from_f(self.alpha);
        self.accumulator += x;
    }
}

impl<'a, SC> PermutationAirBuilder for VerifierConstraintFolder<'a, SC>
where
    SC: StarkGenericConfig,
{
    type MP = ViewPair<'a, SC::Challenge>;

    type RandomVar = SC::Challenge;

    fn permutation(&self) -> Self::MP {
        *self
            .after_challenge
            .first()
            .expect("Challenge phase not supported")
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.challenges
            .first()
            .map(|c| c.as_slice())
            .expect("Challenge phase not supported")
    }
}

impl<'a, SC: StarkGenericConfig> AirBuilderWithPublicValues for VerifierConstraintFolder<'a, SC> {
    type PublicVar = Self::F;

    fn public_values(&self) -> &[Self::F] {
        self.public_values
    }
}

impl<'a, SC> PermutationAirBuilderWithExposedValues for VerifierConstraintFolder<'a, SC>
where
    SC: StarkGenericConfig,
{
    fn permutation_exposed_values(&self) -> &[Self::EF] {
        self.exposed_values_after_challenge
            .first()
            .map(|c| c.as_slice())
            .expect("Challenge phase not supported")
    }
}
