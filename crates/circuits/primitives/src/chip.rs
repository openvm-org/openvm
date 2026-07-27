use openvm_stark_backend::prover::{AirProvingContext, ProverBackend};

/// A chip is a [ProverBackend]-specific object that converts its input into a trace matrix.
///
/// A chip may be stateful and store state on either host or device, although it is preferred that
/// all per-proof state is received through the input.
pub trait Chip<R, PB: ProverBackend> {
    /// Generate all necessary context for proving a single AIR.
    fn generate_proving_ctx(&self, input: R) -> AirProvingContext<PB>;

    /// If this chip always produces a trace with a fixed number of rows (independent of execution),
    /// return that height. Used by metered execution to avoid resetting constant-height chips
    /// on segment boundaries.
    fn constant_trace_height(&self) -> Option<usize> {
        None
    }
}

impl<R, PB: ProverBackend, C: Chip<R, PB> + ?Sized> Chip<R, PB> for std::sync::Arc<C> {
    fn generate_proving_ctx(&self, input: R) -> AirProvingContext<PB> {
        (**self).generate_proving_ctx(input)
    }

    fn constant_trace_height(&self) -> Option<usize> {
        (**self).constant_trace_height()
    }
}
