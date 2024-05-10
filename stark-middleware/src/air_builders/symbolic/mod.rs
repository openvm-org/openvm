// Copied from uni-stark/src/symbolic_builder.rs to allow A: ?Sized

use p3_air::{
    Air, AirBuilder, AirBuilderWithPublicValues, ExtensionBuilder, PairBuilder,
    PermutationAirBuilder,
};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_ceil_usize;
use tracing::instrument;

pub mod symbolic_expression;
pub mod symbolic_variable;

use self::symbolic_expression::SymbolicExpression;
use self::symbolic_variable::{Entry, SymbolicVariable};
use crate::rap::{PermutationAirBuilderWithExposedValues, Rap};

const NUM_PERM_CHALLENGES: usize = 2;
const NUM_PERM_EXPOSED_VALUES: usize = 1;

#[instrument(name = "infer log of constraint degree", skip_all)]
pub fn get_log_quotient_degree<F, A>(
    rap: &A,
    preprocessed_width: usize,
    permutation_width: usize,
    num_public_values: usize,
) -> usize
where
    F: Field,
    A: Rap<SymbolicAirBuilder<F>> + Air<SymbolicAirBuilder<F>> + ?Sized,
{
    // We pad to at least degree 2, since a quotient argument doesn't make sense with smaller degrees.
    let constraint_degree = get_max_constraint_degree(
        rap,
        preprocessed_width,
        permutation_width,
        num_public_values,
    )
    .max(2);

    // The quotient's actual degree is approximately (max_constraint_degree - 1) n,
    // where subtracting 1 comes from division by the zerofier.
    // But we pad it to a power of two so that we can efficiently decompose the quotient.
    log2_ceil_usize(constraint_degree - 1)
}

#[instrument(name = "infer constraint degree", skip_all, level = "debug")]
pub fn get_max_constraint_degree<F, A>(
    rap: &A,
    preprocessed_width: usize,
    permutation_width: usize,
    num_public_values: usize,
) -> usize
where
    F: Field,
    A: Rap<SymbolicAirBuilder<F>> + Air<SymbolicAirBuilder<F>> + ?Sized,
{
    get_symbolic_constraints(
        rap,
        preprocessed_width,
        permutation_width,
        num_public_values,
    )
    .iter()
    .map(|c| c.degree_multiple())
    .max()
    .unwrap_or(0)
}

#[instrument(name = "evaluate constraints symbolically", skip_all, level = "debug")]
pub fn get_symbolic_constraints<F, A>(
    rap: &A,
    preprocessed_width: usize,
    permutation_width: usize,
    num_public_values: usize,
) -> Vec<SymbolicExpression<F>>
where
    F: Field,
    A: Rap<SymbolicAirBuilder<F>> + Air<SymbolicAirBuilder<F>> + ?Sized,
{
    let mut builder = SymbolicAirBuilder::new(
        preprocessed_width,
        rap.width(),
        permutation_width,
        num_public_values,
    );
    Rap::eval(rap, &mut builder);
    builder.constraints()
}

/// An `AirBuilder` for evaluating constraints symbolically, and recording them for later use.
#[derive(Debug)]
pub struct SymbolicAirBuilder<F: Field> {
    preprocessed: RowMajorMatrix<SymbolicVariable<F>>,
    main: RowMajorMatrix<SymbolicVariable<F>>,
    permutation: RowMajorMatrix<SymbolicVariable<F>>,
    public_values: Vec<SymbolicVariable<F>>,
    perm_challenges: Vec<SymbolicVariable<F>>,
    perm_exposed_values: Vec<F>,
    constraints: Vec<SymbolicExpression<F>>,
}

impl<F: Field> SymbolicAirBuilder<F> {
    pub(crate) fn new(
        preprocessed_width: usize,
        main_width: usize,
        permutation_width: usize,
        num_public_values: usize,
    ) -> Self {
        let prep_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..preprocessed_width)
                    .map(move |index| SymbolicVariable::new(Entry::Preprocessed { offset }, index))
            })
            .collect();
        let main_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..main_width)
                    .map(move |index| SymbolicVariable::new(Entry::Main { offset }, index))
            })
            .collect();
        let permutation_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..permutation_width)
                    .map(move |index| SymbolicVariable::new(Entry::Permutation { offset }, index))
            })
            .collect();
        let public_values = (0..num_public_values)
            .map(move |index| SymbolicVariable::new(Entry::Public, index))
            .collect();

        let perm_challenges = (0..NUM_PERM_CHALLENGES)
            .map(move |index| SymbolicVariable::new(Entry::Challenge, index))
            .collect();
        let num_perm_exposed_values = if permutation_width > 0 {
            NUM_PERM_EXPOSED_VALUES
        } else {
            0
        };
        // TODO: This should be a symbolic variable
        let perm_exposed_values = (0..num_perm_exposed_values)
            // .map(move |index| SymbolicVariable::new(Entry::Challenge, index))
            .map(move |_index| F::one())
            .collect();

        Self {
            preprocessed: RowMajorMatrix::new(prep_values, preprocessed_width),
            main: RowMajorMatrix::new(main_values, main_width),
            permutation: RowMajorMatrix::new(permutation_values, permutation_width),
            public_values,
            perm_challenges,
            perm_exposed_values,
            constraints: vec![],
        }
    }

    pub(crate) fn constraints(self) -> Vec<SymbolicExpression<F>> {
        self.constraints
    }
}

impl<F: Field> AirBuilder for SymbolicAirBuilder<F> {
    type F = F;
    type Expr = SymbolicExpression<Self::F>;
    type Var = SymbolicVariable<Self::F>;
    type M = RowMajorMatrix<Self::Var>;

    fn main(&self) -> Self::M {
        self.main.clone()
    }

    fn is_first_row(&self) -> Self::Expr {
        SymbolicExpression::IsFirstRow
    }

    fn is_last_row(&self) -> Self::Expr {
        SymbolicExpression::IsLastRow
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            SymbolicExpression::IsTransition
        } else {
            panic!("uni-stark only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.constraints.push(x.into());
    }
}

impl<F: Field> PairBuilder for SymbolicAirBuilder<F> {
    fn preprocessed(&self) -> Self::M {
        self.preprocessed.clone()
    }
}

impl<F: Field> ExtensionBuilder for SymbolicAirBuilder<F> {
    type EF = F;
    type ExprEF = SymbolicExpression<F>;
    type VarEF = SymbolicVariable<F>;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.constraints.push(x.into());
    }
}

impl<F: Field> PermutationAirBuilder for SymbolicAirBuilder<F> {
    type MP = RowMajorMatrix<Self::VarEF>;
    type RandomVar = SymbolicVariable<F>;

    fn permutation(&self) -> Self::MP {
        self.permutation.clone()
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        &self.perm_challenges
    }
}

impl<F: Field> AirBuilderWithPublicValues for SymbolicAirBuilder<F> {
    type PublicVar = SymbolicVariable<F>;

    fn public_values(&self) -> &[Self::PublicVar] {
        &self.public_values
    }
}

impl<F: Field> PermutationAirBuilderWithExposedValues for SymbolicAirBuilder<F> {
    // TODO: Should this be Self::VarEF?
    fn permutation_exposed_values(&self) -> &[Self::EF] {
        &self.perm_exposed_values
    }
}
