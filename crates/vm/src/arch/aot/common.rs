pub const REG_FOURTH_ARG: &str = "rcx";

pub const REG_C: &str = "rdx";
pub const REG_C_W: &str = "edx";
pub const REG_C_B: &str = "dx";
pub const REG_C_LB: &str = "dl";
pub const REG_THIRD_ARG: &str = "rdx";

pub const REG_B: &str = "rsi";
pub const REG_B_W: &str = "esi";
pub const REG_SECOND_ARG: &str = "rsi";

pub const REG_A: &str = "rdi";
pub const REG_A_W: &str = "edi";
pub const REG_FIRST_ARG: &str = "rdi";

pub const REG_RETURN_VAL: &str = "rax";
pub const REG_D: &str = "rax";
pub const REG_D_W: &str = "eax";
pub const REG_INSTRET_END: &str = "r12";

pub const REG_EXEC_STATE_PTR: &str = "rbx";
pub const REG_TRACE_HEIGHT: &str = "r14";
pub const REG_AS2_PTR: &str = "r15";

pub const DEFAULT_PC_OFFSET: i32 = 4;

pub const RISCV_TO_X86_OVERRIDE_MAP: [Option<&str>; 32] = [
    None,         // x0
    None,         // x1
    None,         // x2
    None,         // x3
    None,         // x4
    None,         // x5
    None,         // x6
    None,         // x7
    None,         // x8
    None,         // x9
    Some("r10d"), // x10
    Some("r11d"), // x11
    Some("r9d"),  // x12
    Some("r8d"),  // x13
    Some("ebp"),  // x14
    Some("r13d"), // x15
    None,         // x16
    None,         // x17
    None,         // x18
    None,         // x19
    None,         // x20
    None,         // x21
    None,         // x22
    None,         // x23
    None,         // x24
    None,         // x25
    None,         // x26
    None,         // x27
    None,         // x28
    None,         // x29
    None,         // x30
    None,         // x31
];

pub fn sync_xmm_to_gpr() -> String {
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

pub fn sync_gpr_to_xmm() -> String {
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

#[derive(Copy, Clone)]
pub enum Width {
    W64,
    W32,
    W16,
    W8L,
    W8H,
}

pub fn convert_x86_reg(any: &str, to: Width) -> Option<&'static str> {
    #[rustfmt::skip]
    const T: [(&str,&str,&str,&str,Option<&str>); 16] = [
        ("rax","eax","ax","al",Some("ah")), ("rbx","ebx","bx","bl",Some("bh")),
        ("rcx","ecx","cx","cl",Some("ch")), ("rdx","edx","dx","dl",Some("dh")),
        ("rsi","esi","si","sil",None),      ("rdi","edi","di","dil",None),
        ("rbp","ebp","bp","bpl",None),      ("rsp","esp","sp","spl",None),
        ("r8","r8d","r8w","r8b",None),      ("r9","r9d","r9w","r9b",None),
        ("r10","r10d","r10w","r10b",None),  ("r11","r11d","r11w","r11b",None),
        ("r12","r12d","r12w","r12b",None),  ("r13","r13d","r13w","r13b",None),
        ("r14","r14d","r14w","r14b",None),  ("r15","r15d","r15w","r15b",None),
    ];

    fn pick(
        row: (
            &'static str,
            &'static str,
            &'static str,
            &'static str,
            Option<&'static str>,
        ),
        w: Width,
    ) -> Option<&'static str> {
        match w {
            Width::W64 => Some(row.0),
            Width::W32 => Some(row.1),
            Width::W16 => Some(row.2),
            Width::W8L => Some(row.3),
            Width::W8H => row.4,
        }
    }

    let key = any.to_ascii_lowercase();
    for row in T {
        if [row.0, row.1, row.2, row.3].iter().any(|&n| n == key) || row.4 == Some(key.as_str()) {
            return pick(row, to);
        }
    }
    None
}
