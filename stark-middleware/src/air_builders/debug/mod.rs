use p3_air::{
    AirBuilder, AirBuilderWithPublicValues, ExtensionBuilder, PairBuilder, PermutationAirBuilder,
};
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::VerticalPair;
use p3_uni_stark::{StarkGenericConfig, Val};

use crate::rap::PermutationAirBuilderWithExposedValues;

pub mod check_constraints;

/// An `AirBuilder` which asserts that each constraint is zero, allowing any failed constraints to
/// be detected early.
pub struct DebugConstraintBuilder<'a, SC: StarkGenericConfig> {
    pub row_index: usize,
    pub preprocessed:
        VerticalPair<RowMajorMatrixView<'a, Val<SC>>, RowMajorMatrixView<'a, Val<SC>>>,
    pub main: VerticalPair<RowMajorMatrixView<'a, Val<SC>>, RowMajorMatrixView<'a, Val<SC>>>,
    pub perm:
        VerticalPair<RowMajorMatrixView<'a, SC::Challenge>, RowMajorMatrixView<'a, SC::Challenge>>,
    pub perm_challenges: &'a [SC::Challenge],
    pub is_first_row: Val<SC>,
    pub is_last_row: Val<SC>,
    pub is_transition: Val<SC>,
    pub public_values: &'a [Val<SC>],
    pub perm_exposed_values: &'a [SC::Challenge],
}

impl<'a, SC> AirBuilder for DebugConstraintBuilder<'a, SC>
where
    SC: StarkGenericConfig,
{
    type F = Val<SC>;
    type Expr = Val<SC>;
    type Var = Val<SC>;
    type M = VerticalPair<RowMajorMatrixView<'a, Val<SC>>, RowMajorMatrixView<'a, Val<SC>>>;

    fn main(&self) -> Self::M {
        self.main
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
        assert_eq!(
            x.into(),
            Val::<SC>::zero(),
            "constraints had nonzero value on row {}",
            self.row_index
        );
    }

    fn assert_eq<I1: Into<Self::Expr>, I2: Into<Self::Expr>>(&mut self, x: I1, y: I2) {
        let x = x.into();
        let y = y.into();
        assert_eq!(
            x, y,
            "values didn't match on row {}: {} != {}",
            self.row_index, x, y
        );
    }
}

impl<'a, SC> PairBuilder for DebugConstraintBuilder<'a, SC>
where
    SC: StarkGenericConfig,
{
    fn preprocessed(&self) -> Self::M {
        self.preprocessed
    }
}

impl<'a, SC> ExtensionBuilder for DebugConstraintBuilder<'a, SC>
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
        assert_eq!(
            x.into(),
            SC::Challenge::zero(),
            "constraints had nonzero value on row {}",
            self.row_index
        );
    }

    fn assert_eq_ext<I1, I2>(&mut self, x: I1, y: I2)
    where
        I1: Into<Self::ExprEF>,
        I2: Into<Self::ExprEF>,
    {
        let x = x.into();
        let y = y.into();
        assert_eq!(
            x, y,
            "values didn't match on row {}: {} != {}",
            self.row_index, x, y
        );
    }
}

impl<'a, SC> PermutationAirBuilder for DebugConstraintBuilder<'a, SC>
where
    SC: StarkGenericConfig,
{
    type MP =
        VerticalPair<RowMajorMatrixView<'a, SC::Challenge>, RowMajorMatrixView<'a, SC::Challenge>>;

    type RandomVar = SC::Challenge;

    fn permutation(&self) -> Self::MP {
        self.perm
    }

    fn permutation_randomness(&self) -> &[Self::EF] {
        self.perm_challenges
    }
}

impl<'a, SC> AirBuilderWithPublicValues for DebugConstraintBuilder<'a, SC>
where
    SC: StarkGenericConfig,
{
    type PublicVar = Val<SC>;

    fn public_values(&self) -> &[Self::F] {
        self.public_values
    }
}

impl<'a, SC> PermutationAirBuilderWithExposedValues for DebugConstraintBuilder<'a, SC>
where
    SC: StarkGenericConfig,
{
    fn permutation_exposed_values(&self) -> &[Self::EF] {
        self.perm_exposed_values
    }
}
