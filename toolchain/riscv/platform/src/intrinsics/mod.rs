pub const CUSTOM_0: u8 = 0x0b;
pub const CUSTOM_1: u8 = 0x2b;

#[macro_export]
macro_rules! custom_insn_i {
    ($opcode:expr, $funct3:expr, $rd:literal, $rs1:literal, $imm:expr) => {
        unsafe {
            core::arch::asm!(concat!(
                ".insn i {opcode}, {funct3}, ",
                $rd,
                ", ",
                $rs1,
                ", {imm}",
            ), opcode = const $opcode, funct3 = const $funct3, imm = const $imm)
        }
    };
    ($opcode:expr, $funct3:expr, $x:ident, $rs1:literal, $imm:expr) => {
        unsafe {
            core::arch::asm!(concat!(
                ".insn i {opcode}, {funct3}, {rd}, ",
                $rs1,
                ", {imm}",
            ), opcode = const $opcode, funct3 = const $funct3, rd = in(reg) $x, imm = const $imm)
        }
    };
}

#[macro_export]
macro_rules! custom_insn_r {
    ($opcode:expr, $funct3:expr, $rd:literal, $rs1:literal, $rs2:literal) => {
        unsafe {
            core::arch::asm!(concat!(
                ".insn r {opcode}, {funct3}, ",
                $rd,
                ", ",
                $rs1,
                ", ",
                $rs2,
            ), opcode = const $opcode, funct3 = const $funct3)
        }
    };
    ($opcode:expr, $funct3:expr, $x:ident, $rs1:literal, $rs2:literal) => {
        unsafe {
            core::arch::asm!(concat!(
                ".insn r {opcode}, {funct3}, {rd}, ",
                $rs1,
                ", ",
                $rs2,
            ), opcode = const $opcode, funct3 = const $funct3, rd = out(reg) $x)
        }
    };
    // TODO: implement more variants with like rs1 = in(reg) $y etc
}
