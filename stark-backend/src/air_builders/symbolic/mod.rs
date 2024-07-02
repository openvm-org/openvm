// Copied from uni-stark/src/symbolic_builder.rs to allow A: ?Sized

use itertools::Itertools;
use p3_air::{
    AirBuilder, AirBuilderWithPublicValues, ExtensionBuilder, PairBuilder, PermutationAirBuilder,
};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_ceil_usize;
use tracing::instrument;

use crate::keygen::types::TraceWidth;
use crate::rap::{PermutationAirBuilderWithExposedValues, Rap};

use super::PartitionedAirBuilder;

use self::symbolic_expression::SymbolicExpression;
use self::symbolic_variable::{Entry, SymbolicVariable};

pub mod symbolic_expression;
pub mod symbolic_variable;

#[instrument(name = "infer log of constraint degree", skip_all)]
pub fn get_log_quotient_degree<F, R>(
    rap: &R,
    width: &TraceWidth,
    num_challenges_to_sample: &[usize],
    num_public_values: usize,
    num_exposed_values_after_challenge: &[usize],
) -> usize
where
    F: Field,
    R: Rap<SymbolicRapBuilder<F>> + ?Sized,
{
    // We pad to at least degree 2, since a quotient argument doesn't make sense with smaller degrees.
    let constraint_degree = get_max_constraint_degree(
        rap,
        width,
        num_challenges_to_sample,
        num_public_values,
        num_exposed_values_after_challenge,
    )
    .max(2);

    // The quotient's actual degree is approximately (max_constraint_degree - 1) n,
    // where subtracting 1 comes from division by the zerofier.
    // But we pad it to a power of two so that we can efficiently decompose the quotient.
    log2_ceil_usize(constraint_degree - 1)
}

#[instrument(name = "infer constraint degree", skip_all, level = "debug")]
pub fn get_max_constraint_degree<F, R>(
    rap: &R,
    width: &TraceWidth,
    num_challenges_to_sample: &[usize],
    num_public_values: usize,
    num_exposed_values_after_challenge: &[usize],
) -> usize
where
    F: Field,
    R: Rap<SymbolicRapBuilder<F>> + ?Sized,
{
    Iterator::max(
        get_symbolic_constraints(
            rap,
            width,
            num_challenges_to_sample,
            num_public_values,
            num_exposed_values_after_challenge,
        )
        .iter()
        .map(|c| c.degree_multiple()),
    )
    .unwrap_or(0)
}

#[instrument(name = "evaluate constraints symbolically", skip_all, level = "debug")]
pub fn get_symbolic_constraints<F, R>(
    rap: &R,
    width: &TraceWidth,
    num_challenges_to_sample: &[usize],
    num_public_values: usize,
    num_exposed_values_after_challenge: &[usize],
) -> Vec<SymbolicExpression<F>>
where
    F: Field,
    R: Rap<SymbolicRapBuilder<F>> + ?Sized,
{
    let mut builder = SymbolicRapBuilder::new(
        width,
        num_challenges_to_sample,
        num_public_values,
        num_exposed_values_after_challenge,
    );
    Rap::eval(rap, &mut builder);
    builder.constraints()
}

/// An `AirBuilder` for evaluating constraints symbolically, and recording them for later use.
#[derive(Debug)]
pub struct SymbolicRapBuilder<F: Field> {
    preprocessed: RowMajorMatrix<SymbolicVariable<F>>,
    partitioned_main: Vec<RowMajorMatrix<SymbolicVariable<F>>>,
    after_challenge: Vec<RowMajorMatrix<SymbolicVariable<F>>>,
    public_values: Vec<SymbolicVariable<F>>,
    challenges: Vec<Vec<SymbolicVariable<F>>>,
    exposed_values_after_challenge: Vec<Vec<SymbolicVariable<F>>>,
    constraints: Vec<SymbolicExpression<F>>,
}

impl<F: Field> SymbolicRapBuilder<F> {
    /// - `num_challenges_to_sample`: for each challenge phase, how many challenges to sample
    /// - `num_exposed_values_after_challenge`: in each challenge phase, how many values to expose to verifier
    pub(crate) fn new(
        width: &TraceWidth,
        num_challenges_to_sample: &[usize],
        num_public_values: usize,
        num_exposed_values_after_challenge: &[usize],
    ) -> Self {
        let preprocessed_width = width.preprocessed.unwrap_or(0);
        let prep_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..width.preprocessed.unwrap_or(0))
                    .map(move |index| SymbolicVariable::new(Entry::Preprocessed { offset }, index))
            })
            .collect();
        let preprocessed = RowMajorMatrix::new(prep_values, preprocessed_width);

        let partitioned_main: Vec<_> = width
            .partitioned_main
            .iter()
            .map(|&width| {
                let mat_values = [0, 1]
                    .into_iter()
                    .flat_map(|offset| {
                        (0..width)
                            .map(move |index| SymbolicVariable::new(Entry::Main { offset }, index))
                    })
                    .collect_vec();
                RowMajorMatrix::new(mat_values, width)
            })
            .collect();
        let after_challenge: Vec<_> = width
            .after_challenge
            .iter()
            .map(|&width| {
                let mat_values = [0, 1]
                    .into_iter()
                    .flat_map(|offset| {
                        (0..width).map(move |index| {
                            SymbolicVariable::new(Entry::Permutation { offset }, index)
                        })
                    })
                    .collect_vec();
                RowMajorMatrix::new(mat_values, width)
            })
            .collect();
        let public_values = (0..num_public_values)
            .map(move |index| SymbolicVariable::new(Entry::Public, index))
            .collect();

        let challenges = num_challenges_to_sample
            .iter()
            .map(|&num_challenges| {
                (0..num_challenges)
                    .map(|index| SymbolicVariable::new(Entry::Challenge, index))
                    .collect_vec()
            })
            .collect_vec();

        let exposed_values_after_challenge = num_exposed_values_after_challenge
            .iter()
            .map(|&num| {
                (0..num)
                    .map(|index| SymbolicVariable::new(Entry::Exposed, index))
                    .collect_vec()
            })
            .collect_vec();

        Self {
            preprocessed,
            partitioned_main,
            after_challenge,
            public_values,
            challenges,
            exposed_values_after_challenge,
            constraints: vec![],
        }
    }

    pub(crate) fn constraints(self) -> Vec<SymbolicExpression<F>> {
        self.constraints
    }
}

impl<F: Field> AirBuilder for SymbolicRapBuilder<F> {
    type F = F;
    type Expr = SymbolicExpression<Self::F>;
    type Var = SymbolicVariable<Self::F>;
    type M = RowMajorMatrix<Self::Var>;

    /// It is difficult to horizontally concatenate matrices when the main trace is partitioned, so we disable this method in that case.
    fn main(&self) -> Self::M {
        if self.partitioned_main.len() == 1 {
            self.partitioned_main[0].clone()
        } else {
            panic!("Main trace is either empty or partitioned. This function should not be used.")
        }
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

impl<F: Field> PairBuilder for SymbolicRapBuilder<F> {
    fn preprocessed(&self) -> Self::M {
        self.preprocessed.clone()
    }
}

impl<F: Field> ExtensionBuilder for SymbolicRapBuilder<F> {
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

impl<F: Field> AirBuilderWithPublicValues for SymbolicRapBuilder<F> {
    type PublicVar = SymbolicVariable<F>;

    fn public_values(&self) -> &[Self::PublicVar] {
        &self.public_values
    }
}

impl<F: Field> PermutationAirBuilder for SymbolicRapBuilder<F> {
    type MP = RowMajorMatrix<Self::VarEF>;
    type RandomVar = SymbolicVariable<F>;

    fn permutation(&self) -> Self::MP {
        self.after_challenge
            .first()
            .expect("Challenge phase not supported")
            .clone()
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.challenges
            .first()
            .map(|c| c.as_slice())
            .expect("Challenge phase not supported")
    }
}

impl<F: Field> PermutationAirBuilderWithExposedValues for SymbolicRapBuilder<F> {
    fn permutation_exposed_values(&self) -> &[Self::VarEF] {
        self.exposed_values_after_challenge
            .first()
            .map(|c| c.as_slice())
            .expect("Challenge phase not supported")
    }
}

impl<F: Field> PartitionedAirBuilder for SymbolicRapBuilder<F> {
    fn partitioned_main(&self) -> &[Self::M] {
        &self.partitioned_main
    }
}
