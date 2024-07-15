use std::iter;

use p3_air::{Air, AirBuilder, PermutationAirBuilder, VirtualPairCol};
use p3_field::Field;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};

/// Interaction debugging tools
pub mod debug;
pub mod rap;
pub mod trace;
mod utils;

#[derive(Copy, Clone, Debug)]
pub enum InteractionType {
    Send,
    Receive,
}

#[derive(Clone, Debug)]
pub struct Interaction<Expr> {
    pub fields: Vec<Expr>,
    pub count: Expr,
    pub bus_index: usize,
    pub interaction_type: InteractionType,
}

/// An [AirBuilder] with additional functionality to build special logUp arguments for
/// communication between AIRs across buses. These arguments use randomness to
/// add additional trace columns (in the extension field) and constraints to the AIR.
pub trait InteractionBuilder: AirBuilder {
    /// Stores a new send interaction in the builder.
    /// `count` can only refer to "local" (current row) variables.
    fn push_send<E: Into<Self::Expr>>(
        &mut self,
        bus_index: usize,
        fields: impl IntoIterator<Item = E>,
        count: impl Into<Self::Expr>,
    ) {
        self.push_interaction(bus_index, fields, count, InteractionType::Send);
    }

    /// Stores a new receive interaction in the builder.
    /// `count` can only refer to "local" (current row) variables.
    fn push_receive<E: Into<Self::Expr>>(
        &mut self,
        bus_index: usize,
        fields: impl IntoIterator<Item = E>,
        count: impl Into<Self::Expr>,
    ) {
        self.push_interaction(bus_index, fields, count, InteractionType::Receive);
    }

    /// Stores a new interaction in the builder.
    /// `count` can only refer to "local" (current row) variables.
    fn push_interaction<E: Into<Self::Expr>>(
        &mut self,
        bus_index: usize,
        fields: impl IntoIterator<Item = E>,
        count: impl Into<Self::Expr>,
        interaction_type: InteractionType,
    );

    /// Returns the current number of interactions.
    fn num_interactions(&self) -> usize;

    /// Returns all interactions stored.
    fn all_interactions(&self) -> &[Interaction<Self::Expr>];

    /// For internal use. For each interaction, the expression for the multiplicity on the _next_ row.
    // This could be supplied by the user, but the devex seems worse.
    fn all_multiplicities_next(&self) -> Vec<Self::Expr>;
}

pub trait AirBridge<F: Field> {
    fn sends(&self) -> Vec<Interaction<F>> {
        vec![]
    }

    fn receives(&self) -> Vec<Interaction<F>> {
        vec![]
    }

    fn all_interactions(&self) -> Vec<(Interaction<F>, InteractionType)> {
        iter::empty()
            .chain(self.sends().into_iter().map(|i| (i, InteractionType::Send)))
            .chain(
                self.receives()
                    .into_iter()
                    .map(|i| (i, InteractionType::Receive)),
            )
            .collect()
    }

    /// Width of the permutation trace.
    fn permutation_width(&self) -> Option<usize> {
        let num_interactions = self.sends().len() + self.receives().len();
        (num_interactions != 0).then_some(num_interactions + 1)
    }
}

/// An interactive AIR is a AIR that can specify buses for sending and receiving data
/// to other AIRs. The original AIR is augmented by virtual columns determined by
/// the interactions to define a [RAP](crate::rap::Rap).
pub trait InteractiveAir<AB: AirBuilder>: Air<AB> + AirBridge<AB::F> {
    /// Generates the permutation trace for the RAP given the main trace.
    /// The permutation trace depends on two random values which the challenger draws
    /// after committing to all parts of the main trace, including multiplicities.
    ///
    /// Returns the permutation trace as a matrix of extension field elements.
    fn generate_permutation_trace(
        &self,
        preprocessed_trace: &Option<RowMajorMatrixView<AB::F>>,
        partitioned_main_trace: &[RowMajorMatrixView<AB::F>],
        permutation_randomness: Option<[AB::EF; 2]>,
    ) -> Option<RowMajorMatrix<AB::EF>>
    where
        AB: PermutationAirBuilder,
    {
        self::trace::generate_permutation_trace(
            self,
            preprocessed_trace,
            partitioned_main_trace,
            permutation_randomness,
        )
    }
}

impl<AB, A> InteractiveAir<AB> for A
where
    AB: AirBuilder,
    A: Air<AB> + AirBridge<AB::F>,
{
}
