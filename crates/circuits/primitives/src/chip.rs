use openvm_stark_backend::prover::{AirProvingContext, ProverBackend};

/// A chip is a [ProverBackend]-specific object that generates a trace matrix.
///
/// A chip may be stateful and store state on either host or device. VM instruction traces receive
/// their per-proof inputs through backend-specific postflight generators instead.
pub trait Chip<PB: ProverBackend> {
    /// Generate all necessary context for proving a single AIR.
    fn generate_proving_ctx(&self) -> AirProvingContext<PB>;

    /// If this chip always produces a trace with a fixed number of rows (independent of execution),
    /// return that height. Used by metered execution to avoid resetting constant-height chips
    /// on segment boundaries.
    fn constant_trace_height(&self) -> Option<usize> {
        None
    }
}

impl<PB: ProverBackend, C: Chip<PB> + ?Sized> Chip<PB> for std::sync::Arc<C> {
    fn generate_proving_ctx(&self) -> AirProvingContext<PB> {
        (**self).generate_proving_ctx()
    }

    fn constant_trace_height(&self) -> Option<usize> {
        (**self).constant_trace_height()
    }
}
