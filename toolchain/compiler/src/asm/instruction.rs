use alloc::{collections::BTreeMap, format};
use core::fmt;

use num_bigint_dig::BigUint;
use p3_field::{ExtensionField, PrimeField32};

use super::A0;

#[derive(Debug, Clone)]
pub enum AsmInstruction<F, EF> {
    /// Load word (dst, src, var_index, size, offset).
    ///
    /// Load a value from the address stored at src(fp) into dst(fp) with given index and offset.
    LoadF(i32, i32, i32, F, F),
    LoadFI(i32, i32, F, F, F),

    /// Store word (val, addr, var_index, size, offset)
    ///
    /// Store a value from val(fp) into the address stored at addr(fp) with given index and offset.
    StoreF(i32, i32, i32, F, F),
    StoreFI(i32, i32, F, F, F),

    /// Set dst = imm.
    ImmF(i32, F),

    /// Copy, dst = src.
    CopyF(i32, i32),

    /// Add, dst = lhs + rhs.
    AddF(i32, i32, i32),

    /// Add immediate, dst = lhs + rhs.
    AddFI(i32, i32, F),

    /// Subtract, dst = lhs - rhs.
    SubF(i32, i32, i32),

    /// Subtract immediate, dst = lhs - rhs.
    SubFI(i32, i32, F),

    /// Subtract value from immediate, dst = lhs - rhs.
    SubFIN(i32, F, i32),

    /// Multiply, dst = lhs * rhs.
    MulF(i32, i32, i32),

    /// Multiply immediate.
    MulFI(i32, i32, F),

    /// Divide, dst = lhs / rhs.
    DivF(i32, i32, i32),

    /// Divide immediate, dst = lhs / rhs.
    DivFI(i32, i32, F),

    /// Divide value from immediate, dst = lhs / rhs.
    DivFIN(i32, F, i32),

    /// U256 equal, dst = lhs == rhs.
    /// (a, b, c) are memory pointers to (*z, *x, *y), which are
    /// themselves memory pointers to (z, x, y) where z = (x == y ? 1 : 0)
    EqU256(i32, i32, i32),

    /// Add extension, dst = lhs + rhs.
    AddE(i32, i32, i32),

    /// Subtract extension, dst = lhs - rhs.
    SubE(i32, i32, i32),

    /// Multiply extension, dst = lhs * rhs.
    MulE(i32, i32, i32),

    /// Divide extension, dst = lhs / rhs.
    DivE(i32, i32, i32),

    ModularAdd(BigUint, i32, i32, i32),
    ModularSub(BigUint, i32, i32, i32),
    ModularMul(BigUint, i32, i32, i32),
    ModularDiv(BigUint, i32, i32, i32),

    /// int add, dst = lhs + rhs.
    Add256(i32, i32, i32),

    /// int subtract, dst = lhs - rhs.
    Sub256(i32, i32, i32),

    /// int multiply, dst = lhs * rhs.
    Mul256(i32, i32, i32),

    /// uint less than, dst = lhs < rhs.
    LessThanU256(i32, i32, i32),

    /// int equal to, dst = lhs == rhs.
    EqualTo256(i32, i32, i32),

    /// int bitwise XOR, dst = lhs ^ rhs
    Xor256(i32, i32, i32),

    /// int bitwise AND, dst = lhs & rhs
    And256(i32, i32, i32),

    /// int bitwise OR, dst = lhs | rhs
    Or256(i32, i32, i32),

    /// signed int less than, dst = lhs < rhs
    LessThanI256(i32, i32, i32),

    /// int shift left, dst = lhs << rhs
    ShiftLeft256(i32, i32, i32),

    /// int shift right logical, dst = lhs >> rhs
    ShiftRightLogic256(i32, i32, i32),

    /// int shift right arithmetic, dst = lhs >> rhs
    ShiftRightArith256(i32, i32, i32),

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

    /// Perform a Poseidon2 permutation on state starting at address `lhs`
    /// and store new state at `rhs`.
    /// (a, b) are pointers to (lhs, rhs).
    Poseidon2Permute(i32, i32),
    /// Perform 2-to-1 cryptographic compression using Poseidon2.
    /// (a, b, c) are memory pointers to (dst, lhs, rhs)
    Poseidon2Compress(i32, i32, i32),

    /// Perform keccak256 hash on variable length byte array input starting at address `src`.
    /// Writes output as array of `u16` limbs to address `dst`.
    /// (a, b, c) are memory pointers to (dst, src, len)
    Keccak256(i32, i32, i32),
    /// Same as `Keccak256`, but with fixed length input (hence length is an immediate value).
    Keccak256FixLen(i32, i32, F),

    /// (dst_ptr_ptr, p_ptr_ptr, q_ptr_ptr) are pointers to pointers to (dst, p, q).
    /// Reads p,q from memory and writes p+q to dst.
    /// Assumes p != +-q as secp256k1 points.
    Secp256k1AddUnequal(i32, i32, i32),
    /// (dst_ptr_ptr, p_ptr_ptr) are pointers to pointers to (dst, p).
    /// Reads p,q from memory and writes 2*p to dst.
    Secp256k1Double(i32, i32),

    /// Print a variable.
    PrintV(i32),

    /// Print a felt.
    PrintF(i32),

    /// Print an extension element.
    PrintE(i32),

    /// Add next input vector to hint stream.
    HintInputVec(),

    /// HintBits(src, len).
    ///
    /// Bit decompose the field element at address `src` to the first `len` little endian bits and add to hint stream.
    HintBits(i32, u32),

    /// HintBytes(src, len).
    ///
    /// Byte decompose the field element at address `src` to the first `len` little endian bytes and add to hint stream.
    HintBytes(i32, u32),

    /// Stores the next hint stream word into value stored at addr + value.
    StoreHintWordI(i32, F),

    /// Publish(val, index).
    Publish(i32, i32),

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
            AsmInstruction::LoadF(dst, src, var_index, size, offset) => {
                write!(
                    f,
                    "lw    ({})fp, ({})fp, ({})fp, {}, {}",
                    dst, src, var_index, size, offset
                )
            }
            AsmInstruction::LoadFI(dst, src, var_index, size, offset) => {
                write!(
                    f,
                    "lwi   ({})fp, ({})fp, {}, {}, {}",
                    dst, src, var_index, size, offset
                )
            }
            AsmInstruction::StoreF(dst, src, var_index, size, offset) => {
                write!(
                    f,
                    "sw    ({})fp, ({})fp, ({})fp, {}, {}",
                    dst, src, var_index, size, offset
                )
            }
            AsmInstruction::StoreFI(dst, src, var_index, size, offset) => {
                write!(
                    f,
                    "swi   ({})fp, ({})fp, {}, {}, {}",
                    dst, src, var_index, size, offset
                )
            }
            AsmInstruction::ImmF(dst, src) => {
                write!(f, "imm   ({})fp, ({})", dst, src)
            }
            AsmInstruction::CopyF(dst, src) => {
                write!(f, "copy  ({})fp, ({})", dst, src)
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
            AsmInstruction::SubFIN(dst, lhs, rhs) => {
                write!(f, "subin ({})fp, {}, ({})fp", dst, lhs, rhs)
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
            AsmInstruction::DivFIN(dst, lhs, rhs) => {
                write!(f, "divi  ({})fp, {}, ({})fp", dst, lhs, rhs)
            }
            AsmInstruction::EqU256(dst, lhs, rhs) => {
                write!(f, "eq  ({})fp, ({})fp, ({})fp", dst, lhs, rhs)
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
            AsmInstruction::DivE(dst, lhs, rhs) => {
                write!(f, "ediv  ({})fp, ({})fp, ({})fp", dst, lhs, rhs)
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
            AsmInstruction::HintBits(src, len) => write!(f, "hint_bits ({})fp, {}", src, len),
            AsmInstruction::HintBytes(src, len) => write!(f, "hint_bytes ({})fp, {}", src, len),
            AsmInstruction::Poseidon2Permute(dst, lhs) => {
                write!(f, "poseidon2_permute ({})fp, ({})fp", dst, lhs)
            }
            AsmInstruction::Poseidon2Compress(result, src1, src2) => {
                write!(
                    f,
                    "poseidon2_compress ({})fp, ({})fp, ({})fp",
                    result, src1, src2
                )
            }
            AsmInstruction::Keccak256(dst, src, len) => {
                write!(f, "keccak256 ({dst})fp, ({src})fp, ({len})fp",)
            }
            AsmInstruction::Keccak256FixLen(dst, src, len) => {
                write!(f, "keccak256 ({dst})fp, ({src})fp, {len}",)
            }
            AsmInstruction::Secp256k1AddUnequal(dst, p, q) => {
                write!(f, "secp256k1_add_unequal ({})fp, ({})fp, ({})fp", dst, p, q)
            }
            AsmInstruction::Secp256k1Double(dst, p) => {
                write!(f, "secp256k1_double ({})fp, ({})fp", dst, p)
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
            AsmInstruction::Publish(val, index) => {
                write!(f, "commit ({})fp ({})fp", val, index)
            }
            AsmInstruction::CycleTrackerStart(name) => {
                write!(f, "cycle_tracker_start {}", name)
            }
            AsmInstruction::CycleTrackerEnd(name) => {
                write!(f, "cycle_tracker_end {}", name)
            }
            AsmInstruction::ModularAdd(modulus, dst, src1, src2) => {
                write!(
                    f,
                    "modular_add with modulus {:?}: ({})fp ({})fp ({})fp",
                    modulus, dst, src1, src2
                )
            }
            AsmInstruction::ModularSub(modulus, dst, src1, src2) => {
                write!(
                    f,
                    "modular_sub with modulus {:?}: ({})fp ({})fp ({})fp",
                    modulus, dst, src1, src2
                )
            }
            AsmInstruction::ModularMul(modulus, dst, src1, src2) => {
                write!(
                    f,
                    "modular_mul with modulus {:?}: ({})fp ({})fp ({})fp",
                    modulus, dst, src1, src2
                )
            }
            AsmInstruction::ModularDiv(modulus, dst, src1, src2) => {
                write!(
                    f,
                    "modular_div with modulus {:?}: ({})fp ({})fp ({})fp",
                    modulus, dst, src1, src2
                )
            }
            AsmInstruction::Add256(dst, src1, src2) => {
                write!(f, "add_256 ({})fp ({})fp ({})fp", dst, src1, src2)
            }
            AsmInstruction::Sub256(dst, src1, src2) => {
                write!(f, "sub_256 ({})fp ({})fp ({})fp", dst, src1, src2)
            }
            AsmInstruction::Mul256(dst, src1, src2) => {
                write!(f, "mul_256 ({})fp ({})fp ({})fp", dst, src1, src2)
            }
            AsmInstruction::LessThanU256(dst, src1, src2) => {
                write!(f, "sltu_256 ({})fp ({})fp ({})fp", dst, src1, src2)
            }
            AsmInstruction::EqualTo256(dst, src1, src2) => {
                write!(f, "eq_256 ({})fp ({})fp ({})fp", dst, src1, src2)
            }
            AsmInstruction::Xor256(dst, src1, src2) => {
                write!(f, "xor_256 ({})fp ({})fp ({})fp", dst, src1, src2)
            }
            AsmInstruction::And256(dst, src1, src2) => {
                write!(f, "and_256 ({})fp ({})fp ({})fp", dst, src1, src2)
            }
            AsmInstruction::Or256(dst, src1, src2) => {
                write!(f, "or_256 ({})fp ({})fp ({})fp", dst, src1, src2)
            }
            AsmInstruction::LessThanI256(dst, src1, src2) => {
                write!(f, "slt_256 ({})fp ({})fp ({})fp", dst, src1, src2)
            }
            AsmInstruction::ShiftLeft256(dst, src1, src2) => {
                write!(f, "sll_256 ({})fp ({})fp ({})fp", dst, src1, src2)
            }
            AsmInstruction::ShiftRightLogic256(dst, src1, src2) => {
                write!(f, "srl_256 ({})fp ({})fp ({})fp", dst, src1, src2)
            }
            AsmInstruction::ShiftRightArith256(dst, src1, src2) => {
                write!(f, "sra_256 ({})fp ({})fp ({})fp", dst, src1, src2)
            }
        }
    }
}
