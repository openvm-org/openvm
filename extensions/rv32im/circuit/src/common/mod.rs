use std::{collections::HashMap, sync::OnceLock};


/*
free x86 registers:
r8
r9
r10
r11
*/
/*
Relative ranking of riscv registers:
- a1
- a0
- a3 
- a2

- a4
- sp
- zero
- a5
- a6
- a7
- ra
- s1
- s0
....
*/

pub const REG_PC: &str = "r13";
pub const REG_PC_W: &str = "r13d";
pub const REG_A: &str = "rcx";
pub const REG_A_W: &str = "eax";
pub const REG_B_W: &str = "ecx";
pub const REG_B_B: &str = "cl";
pub const REG_D_W: &str = "edx";

pub (crate)static RISCV_TO_X86_OVERRIDE_MAP: [Option<&str>; 32] = [ // replace it with string of GPR register, if want to override the default mapping
    None, // x0
    None, // x1
    None, // x2
    None, // x3
    None, // x4
    None, // x5
    None, // x6
    None, // x7
    None, // x8
    None, // x9
    // None,
    // None,
    // None,
    // None,
    Some("r9d"), // x10
    Some("r8d"), // x11
    Some("r11d"), // x12
    Some("r10d"), // x13
    None, // x14
    None, // x15
    None, // x16
    None, // x17
    None, // x18
    None, // x19
    None, // x20
    None, // x21
    None, // x22
    None, // x23
    None, // x24
    None, // x25
    None, // x26
    None, // x27
    None, // x28
    None, // x29
    None, // x30
    None, // x31
];
// return the register of where the corresponding riscv register is stored, AFTER the function call
//  and the string of assembly to get to the register of riscv into `reg_name`; 
pub(crate) fn rv32_register_to_gpr(rv32_reg: u8, gpr: &str) -> String {
    let xmm_map_reg = rv32_reg / 2;
    if rv32_reg % 2 == 0 {
        format!("   pextrd {}, xmm{}, 0\n", gpr, xmm_map_reg)
    } else {
        format!("   pextrd {}, xmm{}, 1\n", gpr, xmm_map_reg)
    }
}

pub(crate) fn gpr_to_rv32_register(gpr: &str, rv32_reg: u8) -> String {
    let xmm_map_reg = rv32_reg / 2;
    if rv32_reg % 2 == 0 {
        format!("   pinsrd xmm{xmm_map_reg}, {gpr}, 0\n")
    } else {
        format!("   pinsrd xmm{xmm_map_reg}, {gpr}, 1\n")
    }
}

pub(crate) fn address_space_start_to_gpr(address_space: u32, gpr: &str) -> String {
    if address_space == 1 {
        if "r15" != gpr {
            return format!("    mov {gpr}, r15\n");
        }
        return "".to_string();
    }

    let xmm_map_reg = match address_space {
        2 => "xmm0",
        3 => "xmm1",
        4 => "xmm2",
        _ => unreachable!("Only address space 1, 2, 3, 4 is supported"),
    };
    format!("   pextrq {gpr}, {xmm_map_reg}, 1\n")
}


/*
pre condition: XMM and GPR registers contain the riscv32 register values
rv32_reg is index of the riscv32 register
gpr: is the TARGET register to write to (GPR)
post condition:
- if rv32_reg is overridden:
    if gpr == override_Reg, data is already in the correct location
    - return empty_string, gpr
    otherwise, write data from override_reg to gpr
- otherwise, copy from associate XMM register to gpr
- 
*/
// if its a temporary register, then even if its overridden, we need to write to it
pub(crate) fn REG_MAPPING_rv32_register_to_gpr(rv32_reg: u8, gpr: &str, is_gpr_force_write: bool) -> (String, String) {
    if let Some(override_reg) = RISCV_TO_X86_OVERRIDE_MAP[rv32_reg as usize] { // a/4 is overridden, b/4 is overridden
        if is_gpr_force_write {
            return (gpr.to_string(), format!("   mov {}, {}\n", gpr, override_reg));
        }
        return (override_reg.to_string(), "".to_string());
    }
    let xmm_map_reg = rv32_reg / 2;
    if rv32_reg % 2 == 0 {
        (gpr.to_string(), format!("   pextrd {}, xmm{}, 0\n", gpr, xmm_map_reg))
    } else {
        (gpr.to_string(), format!("   pextrd {}, xmm{}, 1\n", gpr, xmm_map_reg))
    }
}

// String of assembly to get to the register of riscv into `reg_name`; 
// copy from GPR to rv32_reg
/*
precondition: correct rv32 data is stored in GPR, and needs to be written into the associated rv32 register
postcondition:
- if rv32_reg is overridden:
    if gpr == override_Reg, data is already in the correct location
    - return empty_string
    otherwise, write data from gpr to override_reg
- otherwise, copy from gpr to associate XMM register
*/
pub(crate) fn REG_MAPPING_gpr_to_rv32_register(gpr: &str, rv32_reg: u8) -> String{
    if let Some(override_reg) = RISCV_TO_X86_OVERRIDE_MAP[rv32_reg as usize] {
        if gpr == override_reg { //already in correct location
            return "".to_string();
        }
        return format!("   mov {}, {}\n", override_reg, gpr);
    }
    let xmm_map_reg = rv32_reg / 2;
    if rv32_reg % 2 == 0 {
        format!("   pinsrd xmm{}, {}, 0\n", xmm_map_reg, gpr)
    } else {
        format!("   pinsrd xmm{}, {}, 1\n", xmm_map_reg, gpr)
    }
}


/*
pre condition: XMM stores the ground truth of the registres
post condition: overridden GPR registers also contain the corresponding values
*/
pub(crate) fn SYNC_XMM_TO_GPR() -> String { // these should be saved by caller tho, so can be treated independently
    let mut asm_str = String::new();
    for i in 0..32{
        let xmm_reg = i / 2;
        let lane = i % 2;
        if let Some(override_reg) = RISCV_TO_X86_OVERRIDE_MAP[i] {
            asm_str += &format!("   pextrd {}, xmm{}, {}\n", override_reg, xmm_reg, lane);
            //zero extends the upper bits
        }
    }
    asm_str
}

/*
pre condition: GPR registers contain the correct values, for the associated riscv32 registers
post condition: XMM stores the ground truth of the registres
*/
pub(crate) fn SYNC_GPR_TO_XMM() -> String {
    let mut asm_str = String::new();
    for i in 0..32{
        let xmm_reg = i / 2;
        let lane = i % 2;
        if let Some(override_reg) = RISCV_TO_X86_OVERRIDE_MAP[i] {
            asm_str += &format!("   pinsrd xmm{}, {}, {}\n", xmm_reg, override_reg, lane);
        }
    }
    asm_str
}

