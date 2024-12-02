use strum::FromRepr;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PhantomDiscriminant(pub u16);

/// Phantom instructions owned by the system. These are handled in the `ExecutionSegment`, as opposed to the `PhantomChip`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, FromRepr)]
#[repr(u16)]
pub enum SysPhantom {
    /// Does nothing at constraint and runtime level besides advance pc by [DEFAULT_PC_STEP](super::program::DEFAULT_PC_STEP).
    Nop = 0,
    /// Causes the runtime to panic, on host machine and prints a backtrace.
    DebugPanic,
    /// Start tracing
    CtStart,
    /// End tracing
    CtEnd,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, FromRepr)]
#[repr(u16)]
pub enum NativePhantom {
    /// Native field element print
    Print = 0x10,
    /// Prepare the next input vector for hinting.
    HintInput,
    /// Prepare the little-endian bit decomposition of a variable for hinting.
    HintBits,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, FromRepr)]
#[repr(u16)]
pub enum Rv32Phantom {
    /// Prepare the next input vector for hinting, but prepend it with a 4-byte decomposition of its length instead of one field element.
    HintInput = 0x20,
    /// Peek string from memory and print it to stdout.
    PrintStr,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, FromRepr)]
#[repr(u16)]
pub enum PairingPhantom {
    /// Uses `b` to determine the curve: `b` is the discriminant of `PairingCurve` kind.
    /// Peeks at `[r32{0}(a)..r32{0}(a) + Fp::NUM_LIMBS * 12]_2` to get `f: Fp12` and then resets the hint stream to equal `final_exp_hint(f) = (residue_witness, scaling_factor): (Fp12, Fp12)` as `Fp::NUM_LIMBS * 12 * 2` bytes.
    HintFinalExp = 0x30,
}
