#[macro_export]
macro_rules! custom_insn_i {
    ($opcode:expr, $funct3:expr, $rd:literal, $rs1:literal, $imm:expr) => {
        unsafe {
            asm!(concat!(
                ".insn i {opcode}, {funct3}, ",
                $rd,
                ", ",
                $rs1,
                ", {imm}",
            ), opcode = const $opcode, funct3 = const $funct3, imm = const $imm)
        }
    };
}

#[macro_export]
macro_rules! custom_insn_r {
    ($opcode:expr, $funct3:expr, $rd:literal, $rs1:literal, $rs2:literal) => {
        unsafe {
            asm!(concat!(
                ".insn i {opcode}, {funct3}, ",
                $rd,
                ", ",
                $rs1,
                ", ",
                $rs2,
            ), opcode = const $opcode, funct3 = const $funct3)
        }
    };
}
