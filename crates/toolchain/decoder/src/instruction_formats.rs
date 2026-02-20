// Modified from rrs-lib (https://github.com/GregAC/rrs) on 2026-02-20.
//
// Copyright 2021 Gregory Chadwick <mail@gregchadwick.co.uk>
// Licensed under the Apache License Version 2.0, with LLVM Exceptions, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Structures and constants for instruction decoding
//!
//! The structures directly relate to the RISC-V instruction formats described in the
//! specification. See the [RISC-V specification](https://riscv.org/technical/specifications/) for
//! further details

pub const OPCODE_LOAD: u32 = 0x03;
pub const OPCODE_FENCE: u32 = 0x0f;
pub const OPCODE_OP_IMM: u32 = 0x13;
pub const OPCODE_AUIPC: u32 = 0x17;
pub const OPCODE_OP_IMM_32: u32 = 0x1b;
pub const OPCODE_STORE: u32 = 0x23;
pub const OPCODE_OP: u32 = 0x33;
pub const OPCODE_LUI: u32 = 0x37;
pub const OPCODE_OP_32: u32 = 0x3b;
pub const OPCODE_BRANCH: u32 = 0x63;
pub const OPCODE_JALR: u32 = 0x67;
pub const OPCODE_JAL: u32 = 0x6f;

#[derive(Debug, PartialEq)]
pub struct RType {
    pub funct7: u32,
    pub rs2: usize,
    pub rs1: usize,
    pub funct3: u32,
    pub rd: usize,
}

impl RType {
    pub fn new(insn: u32) -> RType {
        RType {
            funct7: (insn >> 25) & 0x7f,
            rs2: ((insn >> 20) & 0x1f) as usize,
            rs1: ((insn >> 15) & 0x1f) as usize,
            funct3: (insn >> 12) & 0x7,
            rd: ((insn >> 7) & 0x1f) as usize,
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct IType {
    pub imm: i32,
    pub rs1: usize,
    pub funct3: u32,
    pub rd: usize,
}

impl IType {
    pub fn new(insn: u32) -> IType {
        let uimm: i32 = ((insn >> 20) & 0x7ff) as i32;

        let imm: i32 = if (insn & 0x8000_0000) != 0 {
            uimm - (1 << 11)
        } else {
            uimm
        };

        IType {
            imm,
            rs1: ((insn >> 15) & 0x1f) as usize,
            funct3: (insn >> 12) & 0x7,
            rd: ((insn >> 7) & 0x1f) as usize,
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct ITypeShamt {
    pub funct6: u32,
    pub shamt: u32,
    pub rs1: usize,
    pub funct3: u32,
    pub rd: usize,
}

impl ITypeShamt {
    pub fn new(insn: u32) -> ITypeShamt {
        let itype = IType::new(insn);
        let shamt = (itype.imm as u32) & 0x3f;

        ITypeShamt {
            funct6: (insn >> 26) & 0x3f,
            shamt,
            rs1: itype.rs1,
            funct3: itype.funct3,
            rd: itype.rd,
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct SType {
    pub imm: i32,
    pub rs2: usize,
    pub rs1: usize,
    pub funct3: u32,
}

impl SType {
    pub fn new(insn: u32) -> SType {
        let uimm: i32 = (((insn >> 20) & 0x7e0) | ((insn >> 7) & 0x1f)) as i32;

        let imm: i32 = if (insn & 0x8000_0000) != 0 {
            uimm - (1 << 11)
        } else {
            uimm
        };

        SType {
            imm,
            rs2: ((insn >> 20) & 0x1f) as usize,
            rs1: ((insn >> 15) & 0x1f) as usize,
            funct3: (insn >> 12) & 0x7,
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct BType {
    pub imm: i32,
    pub rs2: usize,
    pub rs1: usize,
    pub funct3: u32,
}

impl BType {
    pub fn new(insn: u32) -> BType {
        let uimm: i32 =
            (((insn >> 20) & 0x7e0) | ((insn >> 7) & 0x1e) | ((insn & 0x80) << 4)) as i32;

        let imm: i32 = if (insn & 0x8000_0000) != 0 {
            uimm - (1 << 12)
        } else {
            uimm
        };

        BType {
            imm,
            rs2: ((insn >> 20) & 0x1f) as usize,
            rs1: ((insn >> 15) & 0x1f) as usize,
            funct3: (insn >> 12) & 0x7,
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct UType {
    pub imm: i32,
    pub rd: usize,
}

impl UType {
    pub fn new(insn: u32) -> UType {
        UType {
            imm: (insn & 0xffff_f000) as i32,
            rd: ((insn >> 7) & 0x1f) as usize,
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct JType {
    pub imm: i32,
    pub rd: usize,
}

impl JType {
    pub fn new(insn: u32) -> JType {
        let uimm: i32 =
            ((insn & 0xff000) | ((insn & 0x100000) >> 9) | ((insn >> 20) & 0x7fe)) as i32;

        let imm: i32 = if (insn & 0x8000_0000) != 0 {
            uimm - (1 << 20)
        } else {
            uimm
        };

        JType {
            imm,
            rd: ((insn >> 7) & 0x1f) as usize,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rtype() {
        assert_eq!(
            RType::new(0x0),
            RType {
                funct7: 0,
                rs2: 0,
                rs1: 0,
                funct3: 0,
                rd: 0
            }
        )
    }

    #[test]
    fn test_itype() {
        // addi x23, x31, 2047
        assert_eq!(
            IType::new(0x7fff8b93),
            IType {
                imm: 2047,
                rs1: 31,
                funct3: 0,
                rd: 23
            }
        );

        // addi x23, x31, -1
        assert_eq!(
            IType::new(0xffff8b93),
            IType {
                imm: -1,
                rs1: 31,
                funct3: 0,
                rd: 23
            }
        );

        // addi x23, x31, -2
        assert_eq!(
            IType::new(0xffef8b93),
            IType {
                imm: -2,
                rs1: 31,
                funct3: 0,
                rd: 23
            }
        );

        // ori x13, x7, 4
        assert_eq!(
            IType::new(0x8003e693),
            IType {
                imm: -2048,
                rs1: 7,
                funct3: 0b110,
                rd: 13
            }
        );
    }

    #[test]
    fn test_itype_shamt() {
        // slli x12, x5, 13
        assert_eq!(
            ITypeShamt::new(0x00d29613),
            ITypeShamt {
                funct6: 0,
                shamt: 13,
                rs1: 5,
                funct3: 0b001,
                rd: 12
            }
        );

        // srli x30, x19, 31
        assert_eq!(
            ITypeShamt::new(0x01f9df13),
            ITypeShamt {
                funct6: 0,
                shamt: 31,
                rs1: 19,
                funct3: 0b101,
                rd: 30
            }
        );

        // srai x7, x23, 0
        assert_eq!(
            ITypeShamt::new(0x400bd393),
            ITypeShamt {
                funct6: 0b010000,
                shamt: 0,
                rs1: 23,
                funct3: 0b101,
                rd: 7
            }
        );
    }

    #[test]
    fn test_stype() {
        // sb x31, -2048(x15)
        assert_eq!(
            SType::new(0x81f78023),
            SType {
                imm: -2048,
                rs2: 31,
                rs1: 15,
                funct3: 0,
            }
        );

        // sh x18, 2047(x3)
        assert_eq!(
            SType::new(0x7f219fa3),
            SType {
                imm: 2047,
                rs2: 18,
                rs1: 3,
                funct3: 1,
            }
        );

        // sw x8, 1(x23)
        assert_eq!(
            SType::new(0x008ba0a3),
            SType {
                imm: 1,
                rs2: 8,
                rs1: 23,
                funct3: 2,
            }
        );

        // sw x5, -1(x25)
        assert_eq!(
            SType::new(0xfe5cafa3),
            SType {
                imm: -1,
                rs2: 5,
                rs1: 25,
                funct3: 2,
            }
        );

        // sw x13, 7(x12)
        assert_eq!(
            SType::new(0x00d623a3),
            SType {
                imm: 7,
                rs2: 13,
                rs1: 12,
                funct3: 2,
            }
        );

        // sw x13, -7(x12)
        assert_eq!(
            SType::new(0xfed62ca3),
            SType {
                imm: -7,
                rs2: 13,
                rs1: 12,
                funct3: 2,
            }
        );
    }

    #[test]
    fn test_btype() {
        // beq x10, x14, .-4096
        assert_eq!(
            BType::new(0x80e50063),
            BType {
                imm: -4096,
                rs1: 10,
                rs2: 14,
                funct3: 0b000
            }
        );

        // blt x3, x21, .+4094
        assert_eq!(
            BType::new(0x7f51cfe3),
            BType {
                imm: 4094,
                rs1: 3,
                rs2: 21,
                funct3: 0b100
            }
        );

        // bge x18, x0, .-2
        assert_eq!(
            BType::new(0xfe095fe3),
            BType {
                imm: -2,
                rs1: 18,
                rs2: 0,
                funct3: 0b101
            }
        );

        // bne x15, x16, .+2
        assert_eq!(
            BType::new(0x01079163),
            BType {
                imm: 2,
                rs1: 15,
                rs2: 16,
                funct3: 0b001
            }
        );

        // bgeu x31, x8, .+18
        assert_eq!(
            BType::new(0x008ff963),
            BType {
                imm: 18,
                rs1: 31,
                rs2: 8,
                funct3: 0b111
            }
        );

        // bgeu x31, x8, .-18
        assert_eq!(
            BType::new(0xfe8ff7e3),
            BType {
                imm: -18,
                rs1: 31,
                rs2: 8,
                funct3: 0b111
            }
        );
    }

    #[test]
    fn test_utype() {
        // lui x0, 0xfffff
        assert_eq!(
            UType::new(0xfffff037),
            UType {
                imm: (0xfffff000 as u32) as i32,
                rd: 0,
            }
        );

        // lui x31, 0x0
        assert_eq!(UType::new(0x00000fb7), UType { imm: 0x0, rd: 31 });

        // lui x17, 0x123ab
        assert_eq!(
            UType::new(0x123ab8b7),
            UType {
                imm: 0x123ab000,
                rd: 17,
            }
        );
    }

    #[test]
    fn test_jtype() {
        // jal x0, .+0xffffe
        assert_eq!(
            JType::new(0x7ffff06f),
            JType {
                imm: 0xffffe,
                rd: 0,
            }
        );

        // jal x31, .-0x100000
        assert_eq!(
            JType::new(0x80000fef),
            JType {
                imm: -0x100000,
                rd: 31,
            }
        );

        // jal x13, .-2
        assert_eq!(JType::new(0xfffff6ef), JType { imm: -2, rd: 13 });

        // jal x13, .+2
        assert_eq!(JType::new(0x002006ef), JType { imm: 2, rd: 13 });

        // jal x26, .-46
        assert_eq!(JType::new(0xfd3ffd6f), JType { imm: -46, rd: 26 });

        // jal x26, .+46
        assert_eq!(JType::new(0x02e00d6f), JType { imm: 46, rd: 26 });
    }
}
