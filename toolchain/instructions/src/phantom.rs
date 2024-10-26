use strum::{EnumCount, EnumIter, FromRepr};

/// Enum for different phantom instructions.
/// Phantom instructions affect the runtime of the VM and the trace matrix values.
/// However they all have no AIR constraints besides advancing the pc by [DEFAULT_PC_STEP](super::program::DEFAULT_PC_STEP).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr)]
#[repr(u16)]
pub enum PhantomInstruction {
    /// Does nothing at constraint and runtime level besides advance pc by [DEFAULT_PC_STEP](super::program::DEFAULT_PC_STEP).
    Nop = 0,
    /// Causes the runtime to panic, on host machine and prints a backtrace.
    DebugPanic,
    PrintF,
    /// Prepare the next input vector for hinting.
    HintInput,
    /// Prepare the little-endian bit decomposition of a variable for hinting.
    HintBits,
    /// Start tracing
    CtStart,
    /// End tracing
    CtEnd,
}
