use std::any::Any;

use openvm_stark_backend::prover::{AirProvingContext, ProverBackend};

/// A chip is a [ProverBackend]-specific object that converts execution logs (also referred to as
/// records) into a trace matrix.
///
/// A chip may be stateful and store state on either host or device, although it is preferred that
/// all state is received through records.
pub trait Chip<R, PB: ProverBackend> {
    /// Generate all necessary context for proving a single AIR.
    fn generate_proving_ctx(&self, records: R) -> AirProvingContext<PB>;

    /// If this chip always produces a trace with a fixed number of rows (independent of execution),
    /// return that height. Used by metered execution to avoid resetting constant-height chips
    /// on segment boundaries.
    fn constant_trace_height(&self) -> Option<usize> {
        None
    }
}

/// Auto-implemented trait for downcasting of trait objects.
pub trait AnyChip<R, PB: ProverBackend>: Chip<R, PB> {
    fn as_any(&self) -> &dyn Any;
}

impl<R, PB: ProverBackend, C: Chip<R, PB> + 'static> AnyChip<R, PB> for C {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl<R, PB: ProverBackend, C: Chip<R, PB> + ?Sized> Chip<R, PB> for std::sync::Arc<C> {
    fn generate_proving_ctx(&self, records: R) -> AirProvingContext<PB> {
        (**self).generate_proving_ctx(records)
    }

    fn constant_trace_height(&self) -> Option<usize> {
        (**self).constant_trace_height()
    }
}
