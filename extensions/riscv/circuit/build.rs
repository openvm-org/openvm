#[cfg(feature = "cuda")]
use std::{env, fmt::Write, fs, path::Path};

#[cfg(feature = "cuda")]
use openvm_cuda_builder::{cuda_available, CudaBuilder};
#[cfg(feature = "cuda")]
use openvm_instructions::{LocalOpcode, SystemOpcode, PUBLIC_VALUES_AS};
#[cfg(feature = "cuda")]
use openvm_riscv_transpiler::{
    BaseAluImmOpcode, BaseAluOpcode, BaseAluWImmOpcode, BaseAluWOpcode, BranchEqualOpcode,
    BranchLessThanOpcode, DivRemOpcode, DivRemWOpcode, LessThanImmOpcode, LessThanOpcode,
    MulHOpcode, MulOpcode, MulWOpcode, AuipcOpcode, HintStoreOpcode, JalLuiOpcode,
    JalrOpcode, LoadStoreOpcode, RevealOpcode, ShiftImmOpcode, ShiftOpcode,
    ShiftWImmOpcode, ShiftWOpcode,
};
#[cfg(feature = "cuda")]
fn opcode_family<T: Copy + LocalOpcode>(
    name: &'static str,
    opcodes: &[T],
) -> (&'static str, usize, usize) {
    let first = opcodes
        .first()
        .expect("RV64 replay opcode family must not be empty")
        .global_opcode_usize();
    for (index, opcode) in opcodes.iter().copied().enumerate() {
        assert_eq!(
            opcode.global_opcode_usize(),
            first + index,
            "RV64 replay opcode family must be contiguous"
        );
    }
    (name, first, opcodes.len())
}

#[cfg(feature = "cuda")]
fn opcode<T: LocalOpcode>(name: &'static str, opcode: T) -> (&'static str, usize, usize) {
    (name, opcode.global_opcode_usize(), 1)
}

#[cfg(feature = "cuda")]
fn write_replay_opcode_registry(out_dir: &Path) {
    let families = [
        opcode_family(
            "BASE_ALU",
            &[
                BaseAluOpcode::ADD,
                BaseAluOpcode::SUB,
                BaseAluOpcode::XOR,
                BaseAluOpcode::OR,
                BaseAluOpcode::AND,
            ],
        ),
        opcode_family(
            "SHIFT",
            &[ShiftOpcode::SLL, ShiftOpcode::SRL, ShiftOpcode::SRA],
        ),
        opcode_family("LESS_THAN", &[LessThanOpcode::SLT, LessThanOpcode::SLTU]),
        opcode_family(
            "LOAD_STORE",
            &[
                LoadStoreOpcode::LOADD,
                LoadStoreOpcode::LOADBU,
                LoadStoreOpcode::LOADHU,
                LoadStoreOpcode::LOADWU,
                LoadStoreOpcode::STORED,
                LoadStoreOpcode::STOREW,
                LoadStoreOpcode::STOREH,
                LoadStoreOpcode::STOREB,
                LoadStoreOpcode::LOADB,
                LoadStoreOpcode::LOADH,
                LoadStoreOpcode::LOADW,
            ],
        ),
        opcode_family(
            "BRANCH_EQUAL",
            &[BranchEqualOpcode::BEQ, BranchEqualOpcode::BNE],
        ),
        opcode_family(
            "BRANCH_LESS_THAN",
            &[
                BranchLessThanOpcode::BLT,
                BranchLessThanOpcode::BLTU,
                BranchLessThanOpcode::BGE,
                BranchLessThanOpcode::BGEU,
            ],
        ),
        opcode_family("JAL_LUI", &[JalLuiOpcode::JAL, JalLuiOpcode::LUI]),
        opcode("JALR", JalrOpcode::JALR),
        opcode("AUIPC", AuipcOpcode::AUIPC),
        opcode("MUL", MulOpcode::MUL),
        opcode_family(
            "MULH",
            &[MulHOpcode::MULH, MulHOpcode::MULHSU, MulHOpcode::MULHU],
        ),
        opcode_family(
            "DIVREM",
            &[
                DivRemOpcode::DIV,
                DivRemOpcode::DIVU,
                DivRemOpcode::REM,
                DivRemOpcode::REMU,
            ],
        ),
        opcode_family("BASE_ALU_W", &[BaseAluWOpcode::ADDW, BaseAluWOpcode::SUBW]),
        opcode_family(
            "SHIFT_W",
            &[ShiftWOpcode::SLLW, ShiftWOpcode::SRLW, ShiftWOpcode::SRAW],
        ),
        opcode("MUL_W", MulWOpcode::MULW),
        opcode_family(
            "DIVREM_W",
            &[
                DivRemWOpcode::DIVW,
                DivRemWOpcode::DIVUW,
                DivRemWOpcode::REMW,
                DivRemWOpcode::REMUW,
            ],
        ),
        opcode_family(
            "BASE_ALU_IMM",
            &[
                BaseAluImmOpcode::ADDI,
                BaseAluImmOpcode::XORI,
                BaseAluImmOpcode::ORI,
                BaseAluImmOpcode::ANDI,
            ],
        ),
        opcode_family(
            "SHIFT_IMM",
            &[
                ShiftImmOpcode::SLLI,
                ShiftImmOpcode::SRLI,
                ShiftImmOpcode::SRAI,
            ],
        ),
        opcode_family(
            "LESS_THAN_IMM",
            &[LessThanImmOpcode::SLTI, LessThanImmOpcode::SLTIU],
        ),
        opcode("BASE_ALU_W_IMM", BaseAluWImmOpcode::ADDIW),
        opcode_family(
            "SHIFT_W_IMM",
            &[
                ShiftWImmOpcode::SLLIW,
                ShiftWImmOpcode::SRLIW,
                ShiftWImmOpcode::SRAIW,
            ],
        ),
        opcode_family(
            "HINT_STORE",
            &[
                HintStoreOpcode::HINT_STORED,
                HintStoreOpcode::HINT_BUFFER,
            ],
        ),
        opcode("REVEAL", RevealOpcode::REVEAL),
        opcode("PHANTOM", SystemOpcode::PHANTOM),
        opcode("TERMINATE", SystemOpcode::TERMINATE),
    ];
    let mut header = format!(
        "#pragma once\n\n#include <cstdint>\n\nstatic constexpr uint32_t REVEAL_PUBLIC_VALUES_ADDRESS_SPACE = {PUBLIC_VALUES_AS}u;\n"
    );
    for &(name, base, count) in &families {
        writeln!(
            header,
            "static constexpr uint32_t {name}_OPCODE_BASE = {base}u;\nstatic constexpr uint32_t {name}_OPCODE_COUNT = {count}u;"
        )
        .unwrap();
    }
    fs::write(out_dir.join("checkpoint_replay_opcodes.cuh"), header)
        .expect("write checkpoint replay opcodes");

    let mut rust = String::from("const REPLAY_OPCODES: &[u32] = &[\n");
    for (_, base, count) in families {
        for opcode in base..base + count {
            writeln!(rust, "    {opcode},").unwrap();
        }
    }
    rust.push_str("];\n");
    fs::write(out_dir.join("checkpoint_replay_opcodes.rs"), rust)
        .expect("write checkpoint replay opcode registry");
}

fn main() {
    #[cfg(feature = "cuda")]
    {
        if !cuda_available() {
            return; // Skip CUDA compilation
        }

        let builder = CudaBuilder::new()
            .include_from_dep("DEP_CUDA_COMMON_INCLUDE")
            .include("cuda/include")
            .include("cuda/rvr/include")
            .include("../../../crates/circuits/primitives/cuda/include")
            .include("../../../crates/vm/cuda/include")
            .include("../../../crates/vm/cuda/rvr/include")
            .include("../../riscv-adapters/cuda/include")
            .watch("../../../crates/circuits/primitives/cuda")
            .watch("../../../crates/vm/cuda/rvr/include")
            .watch("../../riscv-adapters/cuda")
            .watch("cuda/include")
            .watch("cuda/rvr")
            .watch("cuda/src")
            .library_name("tracegen_gpu_rv64im")
            .files_from_glob("cuda/src/**/*.cu");

        let out_dir = env::var_os("OUT_DIR").expect("OUT_DIR");
        let out_dir = Path::new(&out_dir);
        write_replay_opcode_registry(out_dir);

        #[cfg(feature = "rvr")]
        let builder = builder
            .include(out_dir)
            .file("cuda/rvr/checkpoint_replay.cu");

        builder.emit_link_directives();
        builder.build();
    }
}
