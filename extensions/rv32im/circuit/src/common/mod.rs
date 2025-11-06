#[cfg(feature = "aot")]
pub use aot::*;

#[cfg(feature = "aot")]
mod aot {
    // Callee saved
    pub const REG_EXEC_STATE_PTR: &str = "rbx";
    pub const REG_INSNS_PTR: &str = "rbp";
    pub const REG_PC: &str = "r13";
    pub const REG_GUEST_MEM_PTR: &str = "r15";

    // Caller saved
    pub const REG_B: &str = "rax";
    pub const REG_B_W: &str = "eax";

    pub const REG_A: &str = "rcx";
    pub const REG_A_W: &str = "ecx";

    pub const REG_FOURTH_ARG: &str = "rcx";
    pub const REG_THIRD_ARG: &str = "rdx";
    pub const REG_SECOND_ARG: &str = "rsi";
    pub const REG_FIRST_ARG: &str = "rdi";
    pub const REG_RETURN_VAL: &str = "rax";

    pub const REG_TMP_W: &str = "r8d";

    pub const REG_C: &str = "r10";
    pub const REG_C_W: &str = "r10d";
    pub const REG_C_B: &str = "r10b";
    pub const REG_AUX: &str = "r11";

    pub const REG_PC_W: &str = "r13d";

    pub const DEFAULT_PC_OFFSET: i32 = 4;

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
            format!("   pinsrd xmm{}, {}, 0\n", xmm_map_reg, gpr)
        } else {
            format!("   pinsrd xmm{}, {}, 1\n", xmm_map_reg, gpr)
        }
    }

    pub(crate) fn address_space_start_to_gpr(address_space: u32, gpr: &str) -> String {
        if address_space == 1 {
            if "r15" != gpr {
                return format!("    mov {}, r15\n", gpr);
            }
            return "".to_string();
        }

        let xmm_map_reg = match address_space {
            2 => "xmm0",
            3 => "xmm1",
            4 => "xmm2",
            _ => unreachable!("Only address space 1, 2, 3, 4 is supported"),
        };
        format!("   pextrq {}, {}, 1\n", gpr, xmm_map_reg)
    }
}
