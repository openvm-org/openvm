use strum::{EnumCount, EnumIter, FromRepr};

pub struct PhantomDiscriminant(pub u16);

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
    /// Prepare the next input vector for hinting, but prepend it with a 4-byte decomposition of its length instead of one field element.
    HintInputRv32,
    /// Prepare the little-endian bit decomposition of a variable for hinting.
    HintBits,
    /// Start tracing
    CtStart,
    /// End tracing
    CtEnd,
    /// Peek string from memory and print it to stdout.
    PrintStrRv32,
    /// Uses `b` to determine the curve: `b` is the discriminant of `PairingCurve` kind.
    /// Peeks at `[r32{0}(a)..r32{0}(a) + Fp::NUM_LIMBS * 12]_2` to get `f: Fp12` and then resets the hint stream to equal `final_exp_hint(f) = (residue_witness, scaling_factor): (Fp12, Fp12)` as `Fp::NUM_LIMBS * 12 * 2` bytes.
    HintFinalExp,
}
