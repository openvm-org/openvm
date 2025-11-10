#[cfg(feature = "aot")]
pub use aot::*;

#[cfg(feature = "aot")]
mod aot {
    // Callee saved
    pub const REG_EXEC_STATE_PTR: &str = "rbx";
    pub const REG_INSNS_PTR: &str = "rbp";

    pub const REG_PC: &str = "r13";
    pub const REG_PC_W: &str = "r13d";

    pub const REG_GUEST_MEM_PTR: &str = "r15";

    // Caller saved
    pub const REG_B: &str = "rax";
    pub const REG_B_W: &str = "eax";
    pub const REG_RETURN_VAL: &str = "rax";

    pub const REG_A: &str = "rcx";
    pub const REG_A_W: &str = "ecx";
    pub const REG_FOURTH_ARG: &str = "rcx";

    pub const REG_THIRD_ARG: &str = "rdx";

    pub const REG_SECOND_ARG: &str = "rsi";
    pub const REG_FIRST_ARG: &str = "rdi";

    pub const REG_TMP_W: &str = "r8d";

    pub const REG_C: &str = "r10";
    pub const REG_C_W: &str = "r10d";
    pub const REG_C_B: &str = "r10b";
    pub const REG_AUX: &str = "r11";

    pub const DEFAULT_PC_OFFSET: i32 = 4;

    pub(crate) use openvm_circuit::arch::aot::common::{
        RISCV_TO_X86_OVERRIDE_MAP, SYNC_GPR_TO_XMM, SYNC_XMM_TO_GPR,
    };

    pub(crate) fn rv32_register_to_gpr(rv32_reg: u8, gpr: &str) -> String {
        let xmm_map_reg = rv32_reg / 2;
        if rv32_reg % 2 == 0 {
            format!("   pextrd {gpr}, xmm{xmm_map_reg}, 0\n")
        } else {
            format!("   pextrd {gpr}, xmm{xmm_map_reg}, 1\n")
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
}
