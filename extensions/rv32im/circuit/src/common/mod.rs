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
