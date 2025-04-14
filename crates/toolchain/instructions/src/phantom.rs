use strum::FromRepr;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PhantomDiscriminant(pub u16);

/// Phantom instructions owned by the system. These are handled in the `ExecutionSegment`, as
/// opposed to the `PhantomChip`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, FromRepr)]
#[repr(u16)]
pub enum SysPhantom {
    /// Does nothing at constraint and runtime level besides advance pc by
    /// [DEFAULT_PC_STEP](super::program::DEFAULT_PC_STEP).
    Nop = 0,
    /// Causes the runtime to panic, on host machine and prints a backtrace.
    DebugPanic,
    /// Start tracing
    CtStart,
    /// End tracing
    CtEnd,
}
