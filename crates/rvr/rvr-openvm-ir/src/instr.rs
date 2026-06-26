use crate::ext::ExtInstr;

/// Register index alias.
pub type Reg = u8;

/// ALU operations shared by R-type and I-type instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AluOp {
    Add,
    Sub,
    Sll,
    Slt,
    Sltu,
    Xor,
    Srl,
    Sra,
    Or,
    And,
}

/// Memory access width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemWidth {
    Byte,
    Half,
    Word,
    Double,
}

impl MemWidth {
    pub fn bytes(self) -> u8 {
        match self {
            MemWidth::Byte => 1,
            MemWidth::Half => 2,
            MemWidth::Word => 4,
            MemWidth::Double => 8,
        }
    }
}

/// Multiply/divide operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MulDivOp {
    Mul,
    Mulh,
    Mulhsu,
    Mulhu,
    Div,
    Divu,
    Rem,
    Remu,
}

/// Body instruction (no control flow). Control flow lives in `Terminator`.
#[derive(Debug, Clone)]
pub enum Instr {
    /// Register-register ALU.
    AluReg {
        op: AluOp,
        rd: Reg,
        rs1: Reg,
        rs2: Reg,
    },
    /// W-suffix register-register ALU (low 32 bits, result sign-extended to 64).
    AluWReg {
        op: AluOp,
        rd: Reg,
        rs1: Reg,
        rs2: Reg,
    },
    /// Register-immediate ALU.
    AluImm {
        op: AluOp,
        rd: Reg,
        rs1: Reg,
        imm: i32,
    },
    /// W-suffix register-immediate ALU (low 32 bits, result sign-extended to 64).
    AluWImm {
        op: AluOp,
        rd: Reg,
        rs1: Reg,
        imm: i32,
    },
    /// Shift by immediate shamt (6-bit for rv64).
    ShiftImm {
        op: AluOp,
        rd: Reg,
        rs1: Reg,
        shamt: u8,
    },
    /// W-suffix shift by immediate shamt (5-bit, result sign-extended to 64).
    ShiftWImm {
        op: AluOp,
        rd: Reg,
        rs1: Reg,
        shamt: u8,
    },
    /// Load upper immediate (rd = imm << 12).
    Lui {
        rd: Reg,
        value: u32,
    },
    /// Add upper immediate to PC (rd = pc + sign_extend(imm << 12), pre-computed).
    Auipc {
        rd: Reg,
        value: u64,
    },
    /// Load from memory.
    Load {
        width: MemWidth,
        signed: bool,
        rd: Reg,
        rs1: Reg,
        offset: i16,
    },
    /// Store to memory.
    Store {
        width: MemWidth,
        rs1: Reg,
        rs2: Reg,
        offset: i16,
    },
    /// Multiply/divide.
    MulDiv {
        op: MulDivOp,
        rd: Reg,
        rs1: Reg,
        rs2: Reg,
    },
    /// W-suffix multiply/divide (low 32 bits, result sign-extended to 64).
    MulDivW {
        op: MulDivOp,
        rd: Reg,
        rs1: Reg,
        rs2: Reg,
    },

    // OpenVM system/IO instructions.
    Nop,
    /// Extension instruction (dispatched via trait object).
    Ext(Box<dyn ExtInstr>),
}

impl Instr {
    pub fn opname(&self) -> &str {
        match self {
            Instr::AluReg { op, .. } => match op {
                AluOp::Add => "add",
                AluOp::Sub => "sub",
                AluOp::Sll => "sll",
                AluOp::Slt => "slt",
                AluOp::Sltu => "sltu",
                AluOp::Xor => "xor",
                AluOp::Srl => "srl",
                AluOp::Sra => "sra",
                AluOp::Or => "or",
                AluOp::And => "and",
            },
            Instr::AluWReg { op, .. } => match op {
                AluOp::Add => "addw",
                AluOp::Sub => "subw",
                AluOp::Sll => "sllw",
                AluOp::Srl => "srlw",
                AluOp::Sra => "sraw",
                _ => unreachable!("no W variant for this alu op"),
            },
            Instr::AluImm { op, .. } => match op {
                AluOp::Add => "addi",
                AluOp::Sub => "subi",
                AluOp::Slt => "slti",
                AluOp::Sltu => "sltiu",
                AluOp::Xor => "xori",
                AluOp::Or => "ori",
                AluOp::And => "andi",
                _ => unreachable!("shift ops use ShiftImm, not AluImm"),
            },
            Instr::AluWImm { op, .. } => match op {
                AluOp::Add => "addiw",
                _ => unreachable!("no W immediate variant for this op"),
            },
            Instr::ShiftImm { op, .. } => match op {
                AluOp::Sll => "slli",
                AluOp::Srl => "srli",
                AluOp::Sra => "srai",
                _ => unreachable!("non-shift ops use AluImm, not ShiftImm"),
            },
            Instr::ShiftWImm { op, .. } => match op {
                AluOp::Sll => "slliw",
                AluOp::Srl => "srliw",
                AluOp::Sra => "sraiw",
                _ => unreachable!("non-shift ops use AluWImm, not ShiftWImm"),
            },
            Instr::Lui { .. } => "lui",
            Instr::Auipc { .. } => "auipc",
            Instr::Load { width, signed, .. } => match (width, signed) {
                (MemWidth::Double, _) => "ld",
                (MemWidth::Word, true) => "lw",
                (MemWidth::Word, false) => "lwu",
                (MemWidth::Half, true) => "lh",
                (MemWidth::Half, false) => "lhu",
                (MemWidth::Byte, true) => "lb",
                (MemWidth::Byte, false) => "lbu",
            },
            Instr::Store { width, .. } => match width {
                MemWidth::Double => "sd",
                MemWidth::Word => "sw",
                MemWidth::Half => "sh",
                MemWidth::Byte => "sb",
            },
            Instr::MulDiv { op, .. } => match op {
                MulDivOp::Mul => "mul",
                MulDivOp::Mulh => "mulh",
                MulDivOp::Mulhsu => "mulhsu",
                MulDivOp::Mulhu => "mulhu",
                MulDivOp::Div => "div",
                MulDivOp::Divu => "divu",
                MulDivOp::Rem => "rem",
                MulDivOp::Remu => "remu",
            },
            Instr::MulDivW { op, .. } => match op {
                MulDivOp::Mul => "mulw",
                MulDivOp::Div => "divw",
                MulDivOp::Divu => "divuw",
                MulDivOp::Rem => "remw",
                MulDivOp::Remu => "remuw",
                _ => unreachable!("no W variant for mul/div ops"),
            },
            Instr::Nop => "nop",
            Instr::Ext(e) => e.opname(),
        }
    }
}
