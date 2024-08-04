use alloc::{collections::BTreeMap, format};
use core::fmt;

use p3_field::{ExtensionField, PrimeField32};

use super::A0;

#[derive(Debug, Clone)]
pub enum AsmInstruction<F, EF> {
    /// Load word (dst, src, offset).
    ///
    /// Load a value from the address stored at src(fp) + offset into dst(fp).
    LoadFI(i32, i32, F),

    /// Store word (val, addr, offset)
    ///
    /// Store a value from val(fp) into the address stored at addr(fp) + offset.
    StoreFI(i32, i32, F),

    /// Add, dst = lhs + rhs.
    AddF(i32, i32, i32),

    /// Add immediate, dst = lhs + rhs.
    AddFI(i32, i32, F),

    /// Subtract, dst = lhs - rhs.
    SubF(i32, i32, i32),

    /// Subtract immediate, dst = lhs - rhs.
    SubFI(i32, i32, F),

    /// Multiply, dst = lhs * rhs.
    MulF(i32, i32, i32),

    /// Multiply immediate.
    MulFI(i32, i32, F),

    /// Divide, dst = lhs / rhs.
    DivF(i32, i32, i32),

    /// Divide immediate, dst = lhs / rhs.
    DivFI(i32, i32, F),

    /// Add extension, dst = lhs + rhs.
    AddE(i32, i32, i32),

    /// Subtract extension, dst = lhs - rhs.
    SubE(i32, i32, i32),

    /// Multiply extension, dst = lhs * rhs.
    MulE(i32, i32, i32),

    /// Extension inverse, dst = 1 / src.
    InvE(i32, i32),

    /// Jump.
    Jump(i32, F),

    /// Branch not equal.
    Bne(F, i32, i32),

    /// Branch not equal immediate.
    BneI(F, i32, F),

    /// Branch equal.
    Beq(F, i32, i32),

    /// Branch equal immediate.
    BeqI(F, i32, F),

    /// Branch not equal extension.
    BneE(F, i32, i32),

    /// Branch not equal immediate extension.
    BneEI(F, i32, EF),

    /// Branch equal extension.
    BeqE(F, i32, i32),

    /// Branch equal immediate extension.
    BeqEI(F, i32, EF),

    /// Trap.
    Trap,

    /// Halt.
    Halt,

    /// Break(label)
    Break(F),

    /// Perform a permutation of the Poseidon2 hash function on the array specified by the ptr.
    Poseidon2Permute(i32, i32),
    Poseidon2Compress(i32, i32, i32),

    /// Print a variable.
    PrintV(i32),

    /// Print a felt.
    PrintF(i32),

    /// Print an extension element.
    PrintE(i32),

    /// Add next input vector to hint stream.
    HintInputVec(),

    /// HintBits(dst, src).
    ///
    /// Bit decompose the field element `src` and add in little endian to hint stream.
    HintBits(i32),

    /// Stores the next hint stream word into value stored at addr + value.
    StoreHintWordI(i32, F),

    /// FRIFold(m, input).
    FriFold(i32, i32),

    /// Publish(val, index).
    Publish(i32, i32),

    /// RegisterPublicValue(val).
    RegisterPublicValue(i32),

    LessThan(i32, i32, i32),

    CycleTrackerStart(String),
    CycleTrackerEnd(String),
}

impl<F: PrimeField32, EF: ExtensionField<F>> AsmInstruction<F, EF> {
    pub fn j(label: F) -> Self {
        AsmInstruction::Jump(A0, label)
    }

    pub fn fmt(&self, labels: &BTreeMap<F, String>, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AsmInstruction::Break(_) => panic!("Unresolved break instruction"),
            AsmInstruction::LessThan(dst, left, right) => {
                write!(f, "lt  ({})fp, {}, {}", dst, left, right,)
            }
            AsmInstruction::LoadFI(dst, src, offset) => {
                write!(f, "lwi   ({})fp, ({})fp, {}", dst, src, offset)
            }
            AsmInstruction::StoreFI(dst, src, offset) => {
                write!(f, "swi   ({})fp, ({})fp, {}", dst, src, offset)
            }
            AsmInstruction::AddF(dst, lhs, rhs) => {
                write!(f, "add   ({})fp, ({})fp, ({})fp", dst, lhs, rhs)
            }
            AsmInstruction::AddFI(dst, lhs, rhs) => {
                write!(f, "addi  ({})fp, ({})fp, {}", dst, lhs, rhs)
            }
            AsmInstruction::SubF(dst, lhs, rhs) => {
                write!(f, "sub   ({})fp, ({})fp, ({})fp", dst, lhs, rhs)
            }
            AsmInstruction::SubFI(dst, lhs, rhs) => {
                write!(f, "subi  ({})fp, ({})fp, {}", dst, lhs, rhs)
            }
            AsmInstruction::MulF(dst, lhs, rhs) => {
                write!(f, "mul   ({})fp, ({})fp, ({})fp", dst, lhs, rhs)
            }
            AsmInstruction::MulFI(dst, lhs, rhs) => {
                write!(f, "muli  ({})fp, ({})fp, {}", dst, lhs, rhs)
            }
            AsmInstruction::DivF(dst, lhs, rhs) => {
                write!(f, "div   ({})fp, ({})fp, ({})fp", dst, lhs, rhs)
            }
            AsmInstruction::DivFI(dst, lhs, rhs) => {
                write!(f, "divi  ({})fp, ({})fp, {}", dst, lhs, rhs)
            }
            AsmInstruction::AddE(dst, lhs, rhs) => {
                write!(f, "eadd ({})fp, ({})fp, ({})fp", dst, lhs, rhs)
            }
            AsmInstruction::SubE(dst, lhs, rhs) => {
                write!(f, "esub  ({})fp, ({})fp, ({})fp", dst, lhs, rhs)
            }
            AsmInstruction::MulE(dst, lhs, rhs) => {
                write!(f, "emul  ({})fp, ({})fp, ({})fp", dst, lhs, rhs)
            }
            AsmInstruction::InvE(dst, src) => {
                write!(f, "einv ({})fp, ({})fp", dst, src)
            }
            AsmInstruction::Jump(dst, label) => {
                write!(
                    f,
                    "j     ({})fp, {}",
                    dst,
                    labels.get(label).unwrap_or(&format!(".L{}", label))
                )
            }
            AsmInstruction::Bne(label, lhs, rhs) => {
                write!(
                    f,
                    "bne   {}, ({})fp, ({})fp",
                    labels.get(label).unwrap_or(&format!(".L{}", label)),
                    lhs,
                    rhs
                )
            }
            AsmInstruction::BneI(label, lhs, rhs) => {
                write!(
                    f,
                    "bnei  {}, ({})fp, {}",
                    labels.get(label).unwrap_or(&format!(".L{}", label)),
                    lhs,
                    rhs
                )
            }
            AsmInstruction::Beq(label, lhs, rhs) => {
                write!(
                    f,
                    "beq  {}, ({})fp, ({})fp",
                    labels.get(label).unwrap_or(&format!(".L{}", label)),
                    lhs,
                    rhs
                )
            }
            AsmInstruction::BeqI(label, lhs, rhs) => {
                write!(
                    f,
                    "beqi {}, ({})fp, {}",
                    labels.get(label).unwrap_or(&format!(".L{}", label)),
                    lhs,
                    rhs
                )
            }
            AsmInstruction::BneE(label, lhs, rhs) => {
                write!(
                    f,
                    "ebne  {}, ({})fp, ({})fp",
                    labels.get(label).unwrap_or(&format!(".L{}", label)),
                    lhs,
                    rhs
                )
            }
            AsmInstruction::BneEI(label, lhs, rhs) => {
                write!(
                    f,
                    "ebnei {}, ({})fp, {}",
                    labels.get(label).unwrap_or(&format!(".L{}", label)),
                    lhs,
                    rhs
                )
            }
            AsmInstruction::BeqE(label, lhs, rhs) => {
                write!(
                    f,
                    "ebeq  {}, ({})fp, ({})fp",
                    labels.get(label).unwrap_or(&format!(".L{}", label)),
                    lhs,
                    rhs
                )
            }
            AsmInstruction::BeqEI(label, lhs, rhs) => {
                write!(
                    f,
                    "ebeqi {}, ({})fp, {}",
                    labels.get(label).unwrap_or(&format!(".L{}", label)),
                    lhs,
                    rhs
                )
            }
            AsmInstruction::Trap => write!(f, "trap"),
            AsmInstruction::Halt => write!(f, "halt"),
            AsmInstruction::HintBits(dst) => write!(f, "hint_bits ({})fp", dst),
            AsmInstruction::Poseidon2Permute(dst, src) => {
                write!(f, "poseidon2_permute ({})fp, ({})fp", dst, src)
            }
            AsmInstruction::PrintF(dst) => {
                write!(f, "print_f ({})fp", dst)
            }
            AsmInstruction::PrintV(dst) => {
                write!(f, "print_v ({})fp", dst)
            }
            AsmInstruction::PrintE(dst) => {
                write!(f, "print_e ({})fp", dst)
            }
            AsmInstruction::HintInputVec() => write!(f, "hint_vec"),
            AsmInstruction::StoreHintWordI(dst, offset) => {
                write!(f, "shintw ({})fp {}", dst, offset)
            }
            AsmInstruction::FriFold(m, input_ptr) => {
                write!(f, "fri_fold ({})fp, ({})fp", m, input_ptr)
            }
            AsmInstruction::Poseidon2Compress(result, src1, src2) => {
                write!(
                    f,
                    "poseidon2_compress ({})fp, {})fp, {})fp",
                    result, src1, src2
                )
            }
            AsmInstruction::Publish(val, index) => {
                write!(f, "commit ({})fp ({})fp", val, index)
            }
            AsmInstruction::RegisterPublicValue(val) => {
                write!(f, "register_public_value ({})fp", val)
            }
            AsmInstruction::CycleTrackerStart(name) => {
                write!(f, "cycle_tracker_start {}", name)
            }
            AsmInstruction::CycleTrackerEnd(name) => {
                write!(f, "cycle_tracker_end {}", name)
            }
        }
    }
}
