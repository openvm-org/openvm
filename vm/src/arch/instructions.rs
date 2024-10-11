use afs_derive::UsizeOpcode;
use strum_macros::{EnumCount, EnumIter, FromRepr};

pub trait UsizeOpcode {
    fn default_offset() -> usize;
    /// Convert from the discriminant of the enum to the typed enum variant.
    /// Default implementation uses `from_repr`.
    fn from_usize(value: usize) -> Self;
    fn as_usize(&self) -> usize;

    fn with_default_offset(&self) -> usize {
        self.as_usize() + Self::default_offset()
    }
}

pub fn with_default_offset<Opcode: UsizeOpcode>(opcode: Opcode) -> usize {
    Opcode::default_offset() + opcode.as_usize()
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum CoreOpcode {
    NOP,
    LOADW,
    STOREW,
    LOADW2,
    STOREW2,
    JAL,
    BEQ,
    BNE,
    TERMINATE,
    PUBLISH,
    FAIL,
    PRINTF,
    /// Instruction to write the next hint word into memory.
    SHINTW,

    // TODO: move these to a separate class, PhantomOpcode or something
    /// Phantom instruction to prepare the next input vector for hinting.
    HINT_INPUT,
    /// Phantom instruction to prepare the little-endian bit decomposition of a variable for hinting.
    HINT_BITS,
    /// Phantom instruction to prepare the little-endian byte decomposition of a variable for hinting.
    HINT_BYTES,
    /// Phantom instruction to start tracing
    CT_START,
    /// Phantom instruction to end tracing
    CT_END,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x100]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum FieldArithmeticOpcode {
    ADD,
    SUB,
    MUL,
    DIV,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x110]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum FieldExtensionOpcode {
    FE4ADD,
    FE4SUB,
    BBE4MUL,
    BBE4DIV,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x170]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum CastfOpcode {
    CASTF,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x120]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Poseidon2Opcode {
    PERM_POS2,
    COMP_POS2,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x130]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Keccak256Opcode {
    KECCAK256,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x140]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum ModularArithmeticOpcode {
    ADD,
    SUB,
    MUL,
    DIV,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x180]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum EccOpcode {
    EC_ADD_NE,
    EC_DOUBLE,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x150]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum U256Opcode {
    // maybe later we will make it uint and specify the parameters in the config
    ADD,
    SUB,
    LT,
    EQ,
    XOR,
    AND,
    OR,
    SLT,

    SLL,
    SRL,
    SRA,

    MUL,
}

impl U256Opcode {
    // Excludes multiplication
    pub fn arithmetic_opcodes() -> impl Iterator<Item = U256Opcode> {
        (U256Opcode::ADD as usize..=U256Opcode::SLT as usize).map(U256Opcode::from_usize)
    }

    pub fn shift_opcodes() -> impl Iterator<Item = U256Opcode> {
        (U256Opcode::SLL as usize..=U256Opcode::SRA as usize).map(U256Opcode::from_usize)
    }
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x160]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum U32Opcode {
    LUI,
    AUIPC,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x300]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum AluOpcode {
    ADD,
    SUB,
    XOR,
    OR,
    AND,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x305]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum ShiftOpcode {
    SLL,
    SRL,
    SRA,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x310]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum LessThanOpcode {
    SLT,
    SLTU,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x320]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Rv32LoadStoreOpcode {
    LOADW,
    STOREW,
    STOREH,
    STOREB,
    LOADB,
    LOADH,
    LOADBU,
    LOADHU,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x330]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum BranchEqualOpcode {
    BEQ,
    BNE,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x335]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum BranchLessThanOpcode {
    BLT,
    BLTU,
    BGE,
    BGEU,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x340]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Rv32JalLuiOpcode {
    JAL,
    LUI,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x345]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Rv32JalrOpcode {
    JALR,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x350]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Rv32AuipcOpcode {
    AUIPC,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x360]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum MulOpcode {
    MUL,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x365]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum MulHOpcode {
    MULH,
    MULHSU,
    MULHU,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x370]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum DivRemOpcode {
    DIV,
    DIVU,
    REM,
    REMU,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0xdeadaf]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum UnimplementedOpcode {
    KECCAK256_RV32,
}
