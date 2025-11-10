pub const RISCV_TO_X86_OVERRIDE_MAP: [Option<&str>; 32] = [
    None,        // x0
    None,        // x1
    None,        // x2
    None,        // x3
    None,        // x4
    None,        // x5
    None,        // x6
    None,        // x7
    None,        // x8
    None,        // x9
    Some("r9d"), // x10
    Some("r8d"), // x11
    None,        // x12
    None,        // x13
    None,        // x14
    None,        // x15
    None,        // x16
    None,        // x17
    None,        // x18
    None,        // x19
    None,        // x20
    None,        // x21
    None,        // x22
    None,        // x23
    None,        // x24
    None,        // x25
    None,        // x26
    None,        // x27
    None,        // x28
    None,        // x29
    None,        // x30
    None,        // x31
];

pub fn SYNC_XMM_TO_GPR() -> String {
    let mut asm_str = String::new();
    for (rv32_reg, override_reg_opt) in RISCV_TO_X86_OVERRIDE_MAP.iter().copied().enumerate() {
        if let Some(override_reg) = override_reg_opt {
            let xmm_reg = rv32_reg / 2;
            let lane = rv32_reg % 2;
            asm_str += &format!("   pextrd {override_reg}, xmm{xmm_reg}, {lane}\n");
        }
    }
    asm_str
}

pub fn SYNC_GPR_TO_XMM() -> String {
    let mut asm_str = String::new();
    for (rv32_reg, override_reg_opt) in RISCV_TO_X86_OVERRIDE_MAP.iter().copied().enumerate() {
        if let Some(override_reg) = override_reg_opt {
            let xmm_reg = rv32_reg / 2;
            let lane = rv32_reg % 2;
            asm_str += &format!("   pinsrd xmm{xmm_reg}, {override_reg}, {lane}\n");
        }
    }
    asm_str
}

