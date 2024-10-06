use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};
use tracing::instrument;

use crate::{prover::USE_DEBUG_BUILDER, rap::AnyRap};

// TODO[osama]: consider moving this to another file
pub struct AirTrace<SC: StarkGenericConfig> {
    pub air: Box<dyn AnyRap<SC>>,
    pub cached_traces: Vec<RowMajorMatrix<Val<SC>>>,
    pub common_trace: RowMajorMatrix<Val<SC>>,
    pub public_values: Vec<Val<SC>>,
}

impl<SC: StarkGenericConfig> AirTrace<SC> {
    pub fn new(
        air: Box<dyn AnyRap<SC>>,
        cached_traces: Vec<RowMajorMatrix<Val<SC>>>,
        common_trace: RowMajorMatrix<Val<SC>>,
        public_values: Vec<Val<SC>>,
    ) -> Self {
        Self {
            air,
            cached_traces,
            common_trace,
            public_values,
        }
    }

    pub fn simple(
        air: Box<dyn AnyRap<SC>>,
        trace: RowMajorMatrix<Val<SC>>,
        public_values: Vec<Val<SC>>,
    ) -> Self {
        Self::new(air, vec![], trace, public_values)
    }

    pub fn simple_no_pis(air: Box<dyn AnyRap<SC>>, trace: RowMajorMatrix<Val<SC>>) -> Self {
        Self::simple(air, trace, vec![])
    }
}

// Copied from valida-util
/// Calculates and returns the multiplicative inverses of each field element, with zero
/// values remaining unchanged.
#[instrument(name = "batch_multiplicative_inverse", level = "info", skip_all)]
pub fn batch_multiplicative_inverse_allowing_zero<F: Field>(values: Vec<F>) -> Vec<F> {
    // Check if values are zero, and construct a new vector with only nonzero values
    let mut nonzero_values = Vec::with_capacity(values.len());
    let mut indices = Vec::with_capacity(values.len());
    for (i, value) in values.iter().cloned().enumerate() {
        if value.is_zero() {
            continue;
        }
        nonzero_values.push(value);
        indices.push(i);
    }

    // Compute the multiplicative inverse of nonzero values
    let inverse_nonzero_values = p3_field::batch_multiplicative_inverse(&nonzero_values);

    // Reconstruct the original vector
    let mut result = values.clone();
    for (i, index) in indices.into_iter().enumerate() {
        result[index] = inverse_nonzero_values[i];
    }

    result
}

/// Disables the debug builder so there are not debug assert panics.
/// Commonly used in negative tests to prevent panics.
pub fn disable_debug_builder() {
    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
}
