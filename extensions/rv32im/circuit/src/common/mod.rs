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
