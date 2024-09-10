// Copied from uni-stark/src/symbolic_builder.rs to allow A: ?Sized

use itertools::Itertools;
use p3_air::{
    AirBuilder, AirBuilderWithPublicValues, ExtensionBuilder, PairBuilder, PermutationAirBuilder,
};
use p3_field::Field;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_util::log2_ceil_usize;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use self::{
    symbolic_expression::SymbolicExpression,
    symbolic_variable::{Entry, SymbolicVariable},
};
use super::PartitionedAirBuilder;
use crate::{
    interaction::{
        Interaction, InteractionBuilder, InteractionType, NUM_PERM_CHALLENGES,
        NUM_PERM_EXPOSED_VALUES,
    },
    keygen::types::{StarkVerifyingParams, TraceWidth},
    rap::{PermutationAirBuilderWithExposedValues, Rap},
};

pub mod symbolic_expression;
pub mod symbolic_variable;

/// Symbolic constraints for a single AIR with interactions.
/// The constraints contain the constraints on the logup partial sums.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub struct SymbolicConstraints<F: Field> {
    /// All constraints of the RAP, including the constraints on the logup partial sums.
    pub constraints: Vec<SymbolicExpression<F>>,
    pub interactions: Vec<Interaction<SymbolicExpression<F>>>,
}

impl<F: Field> SymbolicConstraints<F> {
    pub fn max_constraint_degree(&self) -> usize {
        Iterator::max(self.constraints.iter().map(|c| c.degree_multiple())).unwrap_or(0)
    }

    pub fn get_log_quotient_degree(&self) -> usize {
        // We pad to at least degree 2, since a quotient argument doesn't make sense with smaller degrees.
        let constraint_degree = self.max_constraint_degree().max(2);

        // The quotient's actual degree is approximately (max_constraint_degree - 1) * (trace height),
        // where subtracting 1 comes from division by the zerofier.
        // But we pad it to a power of two so that we can efficiently decompose the quotient.
        log2_ceil_usize(constraint_degree - 1)
    }

    /// Returns the maximum field degree and count degree across all interactions
    pub fn max_interaction_degrees(&self) -> (usize, usize) {
        let max_field_degree = self
            .interactions
            .iter()
            .map(|interaction| {
                interaction
                    .fields
                    .iter()
                    .map(|field| field.degree_multiple())
                    .max()
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0);

        let max_count_degree = self
            .interactions
            .iter()
            .map(|interaction| interaction.count.degree_multiple())
            .max()
            .unwrap_or(0);

        (max_field_degree, max_count_degree)
    }
}

#[instrument(name = "evaluate constraints symbolically", skip_all, level = "debug")]
pub fn get_symbolic_builder<F, R>(
    rap: &R,
    width: &TraceWidth,
    num_public_values: usize,
    num_challenges_to_sample: &[usize],
    num_exposed_values_after_challenge: &[usize],
    interaction_chunk_size: usize,
) -> SymbolicRapBuilder<F>
where
    F: Field,
    R: Rap<SymbolicRapBuilder<F>> + ?Sized,
{
    let mut builder = SymbolicRapBuilder::new(
        width,
        num_public_values,
        num_challenges_to_sample,
        num_exposed_values_after_challenge,
        interaction_chunk_size,
    );
    Rap::eval(rap, &mut builder);
    builder
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
    interactions: Vec<Interaction<SymbolicExpression<F>>>,
    interaction_chunk_size: usize,
}

impl<F: Field> SymbolicRapBuilder<F> {
    /// - `num_challenges_to_sample`: for each challenge phase, how many challenges to sample
    /// - `num_exposed_values_after_challenge`: in each challenge phase, how many values to expose to verifier
    pub(crate) fn new(
        width: &TraceWidth,
        num_public_values: usize,
        num_challenges_to_sample: &[usize],
        num_exposed_values_after_challenge: &[usize],
        interaction_chunk_size: usize,
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
            .enumerate()
            .map(|(part_index, &width)| {
                let mat_values = [0, 1]
                    .into_iter()
                    .flat_map(|offset| {
                        (0..width).map(move |index| {
                            SymbolicVariable::new(Entry::Main { part_index, offset }, index)
                        })
                    })
                    .collect_vec();
                RowMajorMatrix::new(mat_values, width)
            })
            .collect();
        let after_challenge = Self::new_after_challenge(&width.after_challenge);

        let public_values = (0..num_public_values)
            .map(move |index| SymbolicVariable::new(Entry::Public, index))
            .collect();

        let challenges = Self::new_challenges(num_challenges_to_sample);

        let exposed_values_after_challenge =
            Self::new_exposed_values_after_challenge(num_exposed_values_after_challenge);

        Self {
            preprocessed,
            partitioned_main,
            after_challenge,
            public_values,
            challenges,
            exposed_values_after_challenge,
            constraints: vec![],
            interactions: vec![],
            interaction_chunk_size,
        }
    }

    pub fn constraints(self) -> SymbolicConstraints<F> {
        SymbolicConstraints {
            constraints: self.constraints,
            interactions: self.interactions,
        }
    }

    pub fn params(&self) -> StarkVerifyingParams {
        let width = self.width();
        let num_exposed_values_after_challenge = self.num_exposed_values_after_challenge();
        let num_challenges_to_sample = self.num_challenges_to_sample();
        StarkVerifyingParams {
            width,
            num_public_values: self.public_values.len(),
            num_exposed_values_after_challenge,
            num_challenges_to_sample,
        }
    }

    pub fn width(&self) -> TraceWidth {
        let preprocessed_width = self.preprocessed.width();
        TraceWidth {
            preprocessed: (preprocessed_width != 0).then_some(preprocessed_width),
            partitioned_main: self.partitioned_main.iter().map(|m| m.width()).collect(),
            after_challenge: self.after_challenge.iter().map(|m| m.width()).collect(),
        }
    }

    pub fn num_exposed_values_after_challenge(&self) -> Vec<usize> {
        self.exposed_values_after_challenge
            .iter()
            .map(|c| c.len())
            .collect()
    }

    pub fn num_challenges_to_sample(&self) -> Vec<usize> {
        self.challenges.iter().map(|c| c.len()).collect()
    }

    fn new_after_challenge(
        width_after_phase: &[usize],
    ) -> Vec<RowMajorMatrix<SymbolicVariable<F>>> {
        width_after_phase
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
            .collect_vec()
    }

    fn new_challenges(num_challenges_to_sample: &[usize]) -> Vec<Vec<SymbolicVariable<F>>> {
        num_challenges_to_sample
            .iter()
            .map(|&num_challenges| {
                (0..num_challenges)
                    .map(|index| SymbolicVariable::new(Entry::Challenge, index))
                    .collect_vec()
            })
            .collect_vec()
    }

    fn new_exposed_values_after_challenge(
        num_exposed_values_after_challenge: &[usize],
    ) -> Vec<Vec<SymbolicVariable<F>>> {
        num_exposed_values_after_challenge
            .iter()
            .map(|&num| {
                (0..num)
                    .map(|index| SymbolicVariable::new(Entry::Exposed, index))
                    .collect_vec()
            })
            .collect_vec()
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

impl<F: Field> InteractionBuilder for SymbolicRapBuilder<F> {
    fn push_interaction<E: Into<Self::Expr>>(
        &mut self,
        bus_index: usize,
        fields: impl IntoIterator<Item = E>,
        count: impl Into<Self::Expr>,
        interaction_type: InteractionType,
    ) {
        let fields = fields.into_iter().map(|f| f.into()).collect();
        let count = count.into();
        self.interactions.push(Interaction {
            bus_index,
            fields,
            count,
            interaction_type,
        });
    }

    fn num_interactions(&self) -> usize {
        self.interactions.len()
    }

    fn all_interactions(&self) -> &[Interaction<Self::Expr>] {
        &self.interactions
    }

    fn finalize_interactions(&mut self) {
        let num_interactions = self.num_interactions();
        if num_interactions != 0 {
            assert!(
                self.after_challenge.is_empty(),
                "after_challenge width should be auto-populated by the InteractionBuilder"
            );
            assert!(self.challenges.is_empty());
            assert!(self.exposed_values_after_challenge.is_empty());

            let perm_width = (num_interactions + self.interaction_chunk_size - 1)
                / self.interaction_chunk_size
                + 1;
            self.after_challenge = Self::new_after_challenge(&[perm_width]);
            self.challenges = Self::new_challenges(&[NUM_PERM_CHALLENGES]);
            self.exposed_values_after_challenge =
                Self::new_exposed_values_after_challenge(&[NUM_PERM_EXPOSED_VALUES]);
        }
    }

    fn interaction_chunk_size(&self) -> usize {
        self.interaction_chunk_size
    }
}

impl<F: Field> PartitionedAirBuilder for SymbolicRapBuilder<F> {
    fn partitioned_main(&self) -> &[Self::M] {
        &self.partitioned_main
    }
}

#[allow(dead_code)]
struct LocalOnlyChecker;

#[allow(dead_code)]
impl LocalOnlyChecker {
    fn check_var<F: Field>(var: SymbolicVariable<F>) -> bool {
        match var.entry {
            Entry::Preprocessed { offset } => offset == 0,
            Entry::Main { offset, .. } => offset == 0,
            Entry::Permutation { offset } => offset == 0,
            Entry::Public => true,
            Entry::Challenge => true,
            Entry::Exposed => true,
        }
    }

    fn check_expr<F: Field>(expr: &SymbolicExpression<F>) -> bool {
        match expr {
            SymbolicExpression::Variable(var) => Self::check_var(*var),
            SymbolicExpression::IsFirstRow => false,
            SymbolicExpression::IsLastRow => false,
            SymbolicExpression::IsTransition => false,
            SymbolicExpression::Constant(_) => true,
            SymbolicExpression::Add { x, y, .. } => Self::check_expr(x) && Self::check_expr(y),
            SymbolicExpression::Sub { x, y, .. } => Self::check_expr(x) && Self::check_expr(y),
            SymbolicExpression::Neg { x, .. } => Self::check_expr(x),
            SymbolicExpression::Mul { x, y, .. } => Self::check_expr(x) && Self::check_expr(y),
        }
    }
}
