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
}

impl MemWidth {
    pub fn bytes(self) -> u8 {
        match self {
            MemWidth::Byte => 1,
            MemWidth::Half => 2,
            MemWidth::Word => 4,
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
    /// Register-immediate ALU.
    AluImm {
        op: AluOp,
        rd: Reg,
        rs1: Reg,
        imm: i32,
    },
    /// Shift by immediate shamt.
    ShiftImm {
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
    /// Add upper immediate to PC (rd = pc + (imm << 12), pre-computed).
    Auipc {
        rd: Reg,
        value: u32,
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

    // OpenVM system/IO instructions.
    Nop,
    HintInput,
    PrintStr {
        ptr_reg: Reg,
        len_reg: Reg,
    },
    HintRandom {
        num_words_reg: Reg,
    },
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
            Instr::ShiftImm { op, .. } => match op {
                AluOp::Sll => "slli",
                AluOp::Srl => "srli",
                AluOp::Sra => "srai",
                _ => unreachable!("non-shift ops use AluImm, not ShiftImm"),
            },
            Instr::Lui { .. } => "lui",
            Instr::Auipc { .. } => "auipc",
            Instr::Load { width, signed, .. } => match (width, signed) {
                (MemWidth::Word, _) => "lw",
                (MemWidth::Half, true) => "lh",
                (MemWidth::Half, false) => "lhu",
                (MemWidth::Byte, true) => "lb",
                (MemWidth::Byte, false) => "lbu",
            },
            Instr::Store { width, .. } => match width {
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
            Instr::Nop => "nop",
            Instr::HintInput => "hint_input",
            Instr::PrintStr { .. } => "print_str",
            Instr::HintRandom { .. } => "hint_random",
            Instr::Ext(e) => e.opname(),
        }
    }
}
