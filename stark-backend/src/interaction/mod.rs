use p3_air::AirBuilder;
use serde::{Deserialize, Serialize};

use crate::air_builders::symbolic::symbolic_expression::SymbolicExpression;

/// Interaction debugging tools
pub mod debug;
pub mod rap;
pub mod trace;
mod utils;

/// Constants for interactive AIRs
pub const NUM_PERM_CHALLENGES: usize = 2;
pub const NUM_PERM_EXPOSED_VALUES: usize = 1;

#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum InteractionType {
    Send,
    Receive,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Interaction<Expr> {
    pub fields: Vec<Expr>,
    /// The type of `count` is [Expr] but `count` only allows expressions that refer to
    /// "local" (current row) variables and **not** "next" row variables. This is because
    /// the logup constraints involve re-applying the `count` expression on the "next" row.
    // This functionality is implemented by [InteractionBuilder] in the `all_multiplicities_next`
    // method.
    // There is a runtime check during keygen that will panic if this condition is not satisfied:
    // - in `keygen/mod.rs`, `add_partitioned_air`, L153
    // - which in turn calls `SymbolicRapBuilder::push_interaction` in `air_builders/symbolic/mod.rs`, L336
    pub count: Expr,
    pub bus_index: usize,
    pub interaction_type: InteractionType,
}

pub type SymbolicInteraction<F> = Interaction<SymbolicExpression<F>>;

/// An [AirBuilder] with additional functionality to build special logUp arguments for
/// communication between AIRs across buses. These arguments use randomness to
/// add additional trace columns (in the extension field) and constraints to the AIR.
///
/// An interactive AIR is a AIR that can specify buses for sending and receiving data
/// to other AIRs. The original AIR is augmented by virtual columns determined by
/// the interactions to define a [RAP](crate::rap::Rap).
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

    /// For internal use. Called after all constraints prior to challenge phases have been evaluated.
    fn finalize_interactions(&mut self);

    /// Returns number of interactions to bundle in permutation trace
    fn interaction_chunk_size(&self) -> usize;
}
