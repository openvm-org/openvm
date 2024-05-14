use p3_air::{
    AirBuilder, AirBuilderWithPublicValues, ExtensionBuilder, PairBuilder, PermutationAirBuilder,
};
use p3_field::AbstractField;
use p3_uni_stark::{StarkGenericConfig, Val};

use crate::rap::PermutationAirBuilderWithExposedValues;

use super::{PartitionedAirBuilder, ViewPair};

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

impl<'a, SC> PartitionedAirBuilder for VerifierConstraintFolder<'a, SC>
where
    SC: StarkGenericConfig,
{
    fn partitioned_main(&self) -> &[Self::M] {
        &self.partitioned_main
    }
}
