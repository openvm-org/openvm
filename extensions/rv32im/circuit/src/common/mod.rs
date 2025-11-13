#[cfg(feature = "aot")]
pub use aot::*;

#[cfg(feature = "aot")]
mod aot {
    pub(crate) use openvm_circuit::arch::aot::common::{
        sync_gpr_to_xmm, sync_xmm_to_gpr, RISCV_TO_X86_OVERRIDE_MAP,
    };
    pub use openvm_circuit::arch::aot::common::{
        DEFAULT_PC_OFFSET, REG_A, REG_A_W, REG_B, REG_B_W, REG_C, REG_C_B, REG_C_LB, REG_C_W,
        REG_D, REG_D_W, REG_EXEC_STATE_PTR, REG_FIRST_ARG, REG_AS2_PTR, REG_INSNS_PTR,
        REG_PC, REG_PC_W, REG_RETURN_VAL, REG_SECOND_ARG, REG_THIRD_ARG,
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
        if address_space == 2 {
            if "r15" != gpr {
                return format!("    mov {gpr}, r15\n");
            }
            return "".to_string();
        }

        let xmm_map_reg = match address_space {
            1 => "xmm0",
            3 => "xmm1",
            4 => "xmm2",
            _ => unreachable!("Only address space 1, 2, 3, 4 is supported"),
        };
        format!("   pextrq {gpr}, {xmm_map_reg}, 1\n")
    }

    /*
    input:
    - riscv register number
    - gpr register to write into
    - is_gpr_force_write boolean

    output:
    - string representing the general purpose register that stores the value of register number `rv32_reg`
    - emitted assembly string that performs the move
    */
    pub(crate) fn xmm_to_gpr(
        rv32_reg: u8,
        gpr: &str,
        is_gpr_force_write: bool,
    ) -> (String, String) {
        if let Some(override_reg) = RISCV_TO_X86_OVERRIDE_MAP[rv32_reg as usize] {
            // a/4 is overridden, b/4 is overridden
            if is_gpr_force_write {
                return (
                    gpr.to_string(),
                    format!("  mov {}, {}\n", gpr, override_reg),
                );
            }
            return (override_reg.to_string(), "".to_string());
        }
        let xmm_map_reg = rv32_reg / 2;
        if rv32_reg % 2 == 0 {
            (
                gpr.to_string(),
                format!("   pextrd {}, xmm{}, 0\n", gpr, xmm_map_reg),
            )
        } else {
            (
                gpr.to_string(),
                format!("   pextrd {}, xmm{}, 1\n", gpr, xmm_map_reg),
            )
        }
    }

    pub(crate) fn gpr_to_xmm(gpr: &str, rv32_reg: u8) -> String {
        if let Some(override_reg) = RISCV_TO_X86_OVERRIDE_MAP[rv32_reg as usize] {
            if gpr == override_reg {
                //already in correct location
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
