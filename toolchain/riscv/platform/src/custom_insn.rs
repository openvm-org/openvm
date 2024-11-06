//! Macros for adding custom RISC-V instructions in assembly using .insn directives.

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
    ($opcode:expr, $funct3:expr, $x:expr, $rs1:literal, $imm:expr) => {
        unsafe {
            core::arch::asm!(concat!(
                ".insn i {opcode}, {funct3}, {rd}, ",
                $rs1,
                ", {imm}",
            ), opcode = const $opcode, funct3 = const $funct3, rd = in(reg) $x, imm = const $imm)
        }
    };
    ($opcode:expr, $funct3:expr, $x:expr, $y:expr, $imm:expr) => {
        unsafe {
            core::arch::asm!(
                ".insn i {opcode}, {funct3}, {rd}, {rs1}, {imm}",
                opcode = const $opcode, funct3 = const $funct3, rd = in(reg) $x, rs1 = in(reg) $y, imm = const $imm)
        }
    };
}

#[macro_export]
macro_rules! custom_insn_r {
    ($opcode:expr, $funct3:expr, $funct7:expr, $rd:literal, $rs1:literal, $rs2:literal) => {
        unsafe {
            core::arch::asm!(concat!(
                ".insn r {opcode}, {funct3}, {funct7}, ",
                $rd,
                ", ",
                $rs1,
                ", ",
                $rs2,
            ), opcode = const $opcode, funct3 = const $funct3, funct7 = const $funct7)
        }
    };
    ($opcode:expr, $funct3:expr, $funct7:expr, $rd:ident, $rs1:literal, $rs2:literal) => {
        unsafe {
            core::arch::asm!(concat!(
                ".insn r {opcode}, {funct3}, {funct7}, {rd}, ",
                $rs1,
                ", ",
                $rs2,
            ), opcode = const $opcode, funct3 = const $funct3, funct7 = const $funct7, rd = out(reg) $rd)
        }
    };
    ($opcode:expr, $funct3:expr, $funct7:expr, $rd:expr, $rs1:expr, $rs2:expr) => {
        // Note: rd = in(reg) because we expect rd to be a pointer
        unsafe {
            core::arch::asm!(
                ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, {rs2}",
            opcode = const $opcode, funct3 = const $funct3, funct7 = const $funct7, rd = in(reg) $rd, rs1 = in(reg) $rs1, rs2 = in(reg) $rs2)
        }
    };
    // TODO: implement more variants with like rs1 = in(reg) $y etc
}
