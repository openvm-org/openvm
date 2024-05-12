// Folder: Folding builder
use p3_air::{
    AirBuilder, AirBuilderWithPublicValues, ExtensionBuilder, PairBuilder, PermutationAirBuilder,
};
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::VerticalPair;
use p3_uni_stark::{PackedChallenge, PackedVal, StarkGenericConfig, Val};

use crate::rap::PermutationAirBuilderWithExposedValues;

/// A folder for prover constraints.
pub struct ProverConstraintFolder<'a, SC: StarkGenericConfig> {
    pub preprocessed:
        VerticalPair<RowMajorMatrixView<'a, PackedVal<SC>>, RowMajorMatrixView<'a, PackedVal<SC>>>,
    pub partitioned_main: Vec<
        VerticalPair<RowMajorMatrixView<'a, PackedVal<SC>>, RowMajorMatrixView<'a, PackedVal<SC>>>,
    >,
    pub after_challenge: Vec<
        VerticalPair<
            RowMajorMatrixView<'a, PackedChallenge<SC>>,
            RowMajorMatrixView<'a, PackedChallenge<SC>>,
        >,
    >,
    pub challenges: &'a [Vec<PackedChallenge<SC>>],
    pub is_first_row: PackedVal<SC>,
    pub is_last_row: PackedVal<SC>,
    pub is_transition: PackedVal<SC>,
    pub alpha: SC::Challenge,
    pub accumulator: PackedChallenge<SC>,
    pub public_values: &'a [Val<SC>],
    pub exposed_values_after_challenge: &'a [&'a [SC::Challenge]],
}

impl<'a, SC> AirBuilder for ProverConstraintFolder<'a, SC>
where
    SC: StarkGenericConfig,
{
    type F = Val<SC>;
    type Expr = PackedVal<SC>;
    type Var = PackedVal<SC>;
    type M =
        VerticalPair<RowMajorMatrixView<'a, PackedVal<SC>>, RowMajorMatrixView<'a, PackedVal<SC>>>;

    /// It is difficulty to horizontally concatenate matrices when the main trace is partitioned, so we disable this method in that case.
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
            panic!("only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let x: PackedVal<SC> = x.into();
        self.accumulator *= PackedChallenge::<SC>::from_f(self.alpha);
        self.accumulator += x;
    }
}

impl<'a, SC> PairBuilder for ProverConstraintFolder<'a, SC>
where
    SC: StarkGenericConfig,
{
    fn preprocessed(&self) -> Self::M {
        self.preprocessed
    }
}

impl<'a, SC> ExtensionBuilder for ProverConstraintFolder<'a, SC>
where
    SC: StarkGenericConfig,
{
    type EF = SC::Challenge;
    type ExprEF = PackedChallenge<SC>;
    type VarEF = PackedChallenge<SC>;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        let x: PackedChallenge<SC> = x.into();
        self.accumulator *= PackedChallenge::<SC>::from_f(self.alpha);
        self.accumulator += x;
    }
}

impl<'a, SC: StarkGenericConfig> AirBuilderWithPublicValues for ProverConstraintFolder<'a, SC> {
    type PublicVar = Self::F;

    fn public_values(&self) -> &[Self::F] {
        self.public_values
    }
}

// PermutationAirBuilder is just a special kind of RAP builder
impl<'a, SC> PermutationAirBuilder for ProverConstraintFolder<'a, SC>
where
    SC: StarkGenericConfig,
{
    type MP = VerticalPair<
        RowMajorMatrixView<'a, PackedChallenge<SC>>,
        RowMajorMatrixView<'a, PackedChallenge<SC>>,
    >;

    type RandomVar = PackedChallenge<SC>;

    fn permutation(&self) -> Self::MP {
        self.after_challenge
            .get(0)
            .map(|m| *m)
            .unwrap_or(VerticalPair::new(
                RowMajorMatrixView::new(&[], 0),
                RowMajorMatrixView::new(&[], 0),
            ))
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.challenges
            .get(0)
            .map(|c| c.as_slice())
            .unwrap_or(&[] as &[Self::RandomVar])
    }
}

impl<'a, SC> PermutationAirBuilderWithExposedValues for ProverConstraintFolder<'a, SC>
where
    SC: StarkGenericConfig,
{
    fn permutation_exposed_values(&self) -> &[Self::EF] {
        self.exposed_values_after_challenge
            .get(0)
            .map(|c| *c)
            .unwrap_or(&[] as &[Self::EF])
    }
}
