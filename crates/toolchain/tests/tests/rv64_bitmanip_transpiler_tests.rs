//! Transpiler tests for the RISC-V bit-manipulation (Zba/Zbb/Zbs) extension:
//! word → OpenVM instruction mapping, and word ownership between the base
//! RV64IM transpiler extensions and `Rv64BTranspilerExtension`.

use openvm_circuit::{arch::VmExecutor, utils::air_test};
use openvm_instructions::{
    exe::VmExe, instruction::Instruction, program::Program, riscv::RV64_REGISTER_NUM_LIMBS,
    LocalOpcode, SystemOpcode,
};
use openvm_riscv_circuit::{Rv64ImBBuilder, Rv64ImBConfig};
use openvm_riscv_transpiler::{
    BitwiseInvOpcode, ByteUnaryOpcode, CountZerosOpcode, CountZerosWOpcode, CpopOpcode,
    CpopWOpcode, MinMaxOpcode, RotateImmOpcode, RotateOpcode, RotateWImmOpcode, RotateWOpcode,
    Rv64BTranspilerExtension, Rv64ITranspilerExtension, Rv64IoTranspilerExtension,
    Rv64MTranspilerExtension, ShAddOpcode, SingleBitImmOpcode, SingleBitOpcode, SlliUwOpcode,
};
use openvm_stark_sdk::{openvm_stark_backend::p3_field::PrimeField32, p3_baby_bear::BabyBear};
use openvm_transpiler::transpiler::Transpiler;

type F = BabyBear;

fn rv64im_transpiler() -> Transpiler<F> {
    Transpiler::<F>::default()
        .with_extension(Rv64ITranspilerExtension)
        .with_extension(Rv64MTranspilerExtension)
        .with_extension(Rv64IoTranspilerExtension)
}

fn rv64imb_transpiler() -> Transpiler<F> {
    rv64im_transpiler().with_extension(Rv64BTranspilerExtension)
}

/// Expected operand layout of a transpiled bit-manipulation instruction.
enum Operands {
    /// Register-register: `c` = rs2 pointer, `e` = 1 (register address space).
    Reg { rd: u32, rs1: u32, rs2: u32 },
    /// Shift-immediate: `c` = shamt, `e` = 0 (immediate).
    Shamt { rd: u32, rs1: u32, shamt: u32 },
    /// Unary: `c` = 0, `e` = 0.
    Unary { rd: u32, rs1: u32 },
}

/// Golden encodings generated with
/// `llvm-mc -triple=riscv64 -mattr=+zba,+zbb,+zbs -show-encoding` (LLVM 22),
/// i.e. the exact words the guest compiler emits.
#[rustfmt::skip]
fn golden_cases() -> Vec<(&'static str, u32, usize, Operands)> {
    vec![
        ("add.uw",    0x087302bb, ShAddOpcode::ADD_UW.global_opcode().as_usize(),        Operands::Reg { rd: 5, rs1: 6, rs2: 7 }),
        ("sh1add",    0x20a4a433, ShAddOpcode::SH1ADD.global_opcode().as_usize(),        Operands::Reg { rd: 8, rs1: 9, rs2: 10 }),
        ("sh2add",    0x20d645b3, ShAddOpcode::SH2ADD.global_opcode().as_usize(),        Operands::Reg { rd: 11, rs1: 12, rs2: 13 }),
        ("sh3add",    0x2107e733, ShAddOpcode::SH3ADD.global_opcode().as_usize(),        Operands::Reg { rd: 14, rs1: 15, rs2: 16 }),
        ("sh1add.uw", 0x213928bb, ShAddOpcode::SH1ADD_UW.global_opcode().as_usize(),     Operands::Reg { rd: 17, rs1: 18, rs2: 19 }),
        ("sh2add.uw", 0x216aca3b, ShAddOpcode::SH2ADD_UW.global_opcode().as_usize(),     Operands::Reg { rd: 20, rs1: 21, rs2: 22 }),
        ("sh3add.uw", 0x219c6bbb, ShAddOpcode::SH3ADD_UW.global_opcode().as_usize(),     Operands::Reg { rd: 23, rs1: 24, rs2: 25 }),
        ("slli.uw",   0x0a5d9d1b, SlliUwOpcode::SLLI_UW.global_opcode().as_usize(),      Operands::Shamt { rd: 26, rs1: 27, shamt: 37 }),
        ("andn",      0x407372b3, BitwiseInvOpcode::ANDN.global_opcode().as_usize(),     Operands::Reg { rd: 5, rs1: 6, rs2: 7 }),
        ("orn",       0x40a4e433, BitwiseInvOpcode::ORN.global_opcode().as_usize(),      Operands::Reg { rd: 8, rs1: 9, rs2: 10 }),
        ("xnor",      0x40d645b3, BitwiseInvOpcode::XNOR.global_opcode().as_usize(),     Operands::Reg { rd: 11, rs1: 12, rs2: 13 }),
        ("clz",       0x60079713, CountZerosOpcode::CLZ.global_opcode().as_usize(),      Operands::Unary { rd: 14, rs1: 15 }),
        ("ctz",       0x60189813, CountZerosOpcode::CTZ.global_opcode().as_usize(),      Operands::Unary { rd: 16, rs1: 17 }),
        ("cpop",      0x60299913, CpopOpcode::CPOP.global_opcode().as_usize(),           Operands::Unary { rd: 18, rs1: 19 }),
        ("clzw",      0x600a9a1b, CountZerosWOpcode::CLZW.global_opcode().as_usize(),    Operands::Unary { rd: 20, rs1: 21 }),
        ("ctzw",      0x601b9b1b, CountZerosWOpcode::CTZW.global_opcode().as_usize(),    Operands::Unary { rd: 22, rs1: 23 }),
        ("cpopw",     0x602c9c1b, CpopWOpcode::CPOPW.global_opcode().as_usize(),         Operands::Unary { rd: 24, rs1: 25 }),
        ("min",       0x0a7342b3, MinMaxOpcode::MIN.global_opcode().as_usize(),          Operands::Reg { rd: 5, rs1: 6, rs2: 7 }),
        ("minu",      0x0aa4d433, MinMaxOpcode::MINU.global_opcode().as_usize(),         Operands::Reg { rd: 8, rs1: 9, rs2: 10 }),
        ("max",       0x0ad665b3, MinMaxOpcode::MAX.global_opcode().as_usize(),          Operands::Reg { rd: 11, rs1: 12, rs2: 13 }),
        ("maxu",      0x0b07f733, MinMaxOpcode::MAXU.global_opcode().as_usize(),         Operands::Reg { rd: 14, rs1: 15, rs2: 16 }),
        ("sext.b",    0x60491893, ByteUnaryOpcode::SEXT_B.global_opcode().as_usize(),    Operands::Unary { rd: 17, rs1: 18 }),
        ("sext.h",    0x605a1993, ByteUnaryOpcode::SEXT_H.global_opcode().as_usize(),    Operands::Unary { rd: 19, rs1: 20 }),
        ("zext.h",    0x080b4abb, ByteUnaryOpcode::ZEXT_H.global_opcode().as_usize(),    Operands::Unary { rd: 21, rs1: 22 }),
        ("rol",       0x607312b3, RotateOpcode::ROL.global_opcode().as_usize(),          Operands::Reg { rd: 5, rs1: 6, rs2: 7 }),
        ("ror",       0x60a4d433, RotateOpcode::ROR.global_opcode().as_usize(),          Operands::Reg { rd: 8, rs1: 9, rs2: 10 }),
        ("rori",      0x62d65593, RotateImmOpcode::RORI.global_opcode().as_usize(),      Operands::Shamt { rd: 11, rs1: 12, shamt: 45 }),
        ("rolw",      0x60f716bb, RotateWOpcode::ROLW.global_opcode().as_usize(),        Operands::Reg { rd: 13, rs1: 14, rs2: 15 }),
        ("rorw",      0x6128d83b, RotateWOpcode::RORW.global_opcode().as_usize(),        Operands::Reg { rd: 16, rs1: 17, rs2: 18 }),
        ("roriw",     0x615a599b, RotateWImmOpcode::RORIW.global_opcode().as_usize(),    Operands::Shamt { rd: 19, rs1: 20, shamt: 21 }),
        ("orc.b",     0x287bdb13, ByteUnaryOpcode::ORC_B.global_opcode().as_usize(),     Operands::Unary { rd: 22, rs1: 23 }),
        ("rev8",      0x6b8cdc13, ByteUnaryOpcode::REV8.global_opcode().as_usize(),      Operands::Unary { rd: 24, rs1: 25 }),
        ("bclr",      0x487312b3, SingleBitOpcode::BCLR.global_opcode().as_usize(),      Operands::Reg { rd: 5, rs1: 6, rs2: 7 }),
        ("bset",      0x28a49433, SingleBitOpcode::BSET.global_opcode().as_usize(),      Operands::Reg { rd: 8, rs1: 9, rs2: 10 }),
        ("binv",      0x68d615b3, SingleBitOpcode::BINV.global_opcode().as_usize(),      Operands::Reg { rd: 11, rs1: 12, rs2: 13 }),
        ("bext",      0x4907d733, SingleBitOpcode::BEXT.global_opcode().as_usize(),      Operands::Reg { rd: 14, rs1: 15, rs2: 16 }),
        ("bclri",     0x4af91893, SingleBitImmOpcode::BCLRI.global_opcode().as_usize(),  Operands::Shamt { rd: 17, rs1: 18, shamt: 47 }),
        ("bseti",     0x2b0a1993, SingleBitImmOpcode::BSETI.global_opcode().as_usize(),  Operands::Shamt { rd: 19, rs1: 20, shamt: 48 }),
        ("binvi",     0x6b1b1a93, SingleBitImmOpcode::BINVI.global_opcode().as_usize(),  Operands::Shamt { rd: 21, rs1: 22, shamt: 49 }),
        ("bexti",     0x4b2c5b93, SingleBitImmOpcode::BEXTI.global_opcode().as_usize(),  Operands::Shamt { rd: 23, rs1: 24, shamt: 50 }),
    ]
}

#[test]
fn transpiles_bitmanip_golden_words() {
    let transpiler = rv64imb_transpiler();
    for (name, word, expected_opcode, operands) in golden_cases() {
        let program = transpiler
            .transpile(&[word])
            .unwrap_or_else(|e| panic!("{name}: transpilation failed: {e}"));
        assert_eq!(program.len(), 1, "{name}: expected exactly one instruction");
        let insn = program[0].as_ref().unwrap_or_else(|| panic!("{name}: gap"));

        assert_eq!(
            insn.opcode.as_usize(),
            expected_opcode,
            "{name}: wrong opcode"
        );
        let limbs = RV64_REGISTER_NUM_LIMBS as u32;
        let (rd, rs1, c, e) = match operands {
            Operands::Reg { rd, rs1, rs2 } => (rd, rs1, rs2 * limbs, 1),
            Operands::Shamt { rd, rs1, shamt } => (rd, rs1, shamt, 0),
            Operands::Unary { rd, rs1 } => (rd, rs1, 0, 0),
        };
        assert_eq!(insn.a.as_canonical_u32(), rd * limbs, "{name}: wrong a/rd");
        assert_eq!(
            insn.b.as_canonical_u32(),
            rs1 * limbs,
            "{name}: wrong b/rs1"
        );
        assert_eq!(insn.c.as_canonical_u32(), c, "{name}: wrong c operand");
        assert_eq!(insn.d.as_canonical_u32(), 1, "{name}: wrong d operand");
        assert_eq!(insn.e.as_canonical_u32(), e, "{name}: wrong e operand");
    }
}

#[test]
fn bitmanip_words_trap_without_b_extension() {
    // Without `Rv64BTranspilerExtension`, the base RV64IM extensions must not
    // claim bit-manipulation words (no misdecode into a base instruction);
    // they become `unimp` (TERMINATE with exit code 2) like any other
    // unsupported instruction.
    let transpiler = rv64im_transpiler();
    for (name, word, _, _) in golden_cases() {
        let program = transpiler
            .transpile(&[word])
            .unwrap_or_else(|e| panic!("{name}: transpilation failed: {e}"));
        let insn = program[0].as_ref().unwrap_or_else(|| panic!("{name}: gap"));
        assert_eq!(
            insn.opcode,
            SystemOpcode::TERMINATE.global_opcode(),
            "{name}: expected unimp without the B extension"
        );
        assert_eq!(insn.c.as_canonical_u32(), 2, "{name}: expected exit code 2");
    }
}

#[test]
fn base_words_unaffected_by_b_extension() {
    // Base RV64IM words must transpile identically with and without the B
    // extension registered (the ownership guard must not over-exclude).
    let words = [
        0x003100b3, // add x1, x2, x3
        0x403100b3, // sub x1, x2, x3
        0x0033a0b3, // slt x1, x7, x3
        0x40335313, // srai x6, x6, 3
        0x0023929b, // slliw x5, x7, 2
        0x02310133, // mul x2, x2, x3
    ];
    let with_b = rv64imb_transpiler();
    let without_b = rv64im_transpiler();
    for word in words {
        let a = with_b.transpile(&[word]).unwrap();
        let b = without_b.transpile(&[word]).unwrap();
        assert_eq!(a, b, "word {word:#010x} transpiles differently with B");
        assert_ne!(
            a[0].as_ref().unwrap().opcode,
            SystemOpcode::TERMINATE.global_opcode(),
            "word {word:#010x} unexpectedly unsupported"
        );
    }
}

#[test]
fn bitmanip_rd_zero_transpiles_to_nop() {
    // Writes to x0 are architectural nops; the B extension claims the word and
    // emits the canonical nop (PHANTOM), exactly like the base extensions.
    let transpiler = rv64imb_transpiler();
    for (name, word, _, _) in golden_cases() {
        let word_rd0 = word & !(0x1f << 7);
        let program = transpiler
            .transpile(&[word_rd0])
            .unwrap_or_else(|e| panic!("{name} (rd=x0): transpilation failed: {e}"));
        let insn = program[0]
            .as_ref()
            .unwrap_or_else(|| panic!("{name} (rd=x0): gap"));
        assert_eq!(
            insn.opcode,
            SystemOpcode::PHANTOM.global_opcode(),
            "{name} (rd=x0): expected nop"
        );
    }
}

#[test]
fn mixed_base_and_bitmanip_program() {
    // A program interleaving base and B words keeps 1:1 word→instruction
    // alignment (PC mapping) and every word is claimed by exactly one
    // extension (no AmbiguousNextInstruction).
    let words = [
        0x003100b3, // add x1, x2, x3
        0x20a4a433, // sh1add x8, x9, x10
        0x02310133, // mul x2, x2, x3
        0x62d65593, // rori x11, x12, 45
        0x60079713, // clz x14, x15
        0x0033a0b3, // slt x1, x7, x3
    ];
    let program = rv64imb_transpiler().transpile(&words).unwrap();
    assert_eq!(program.len(), words.len());
    let expected = [
        None, // base op, checked as "not terminate" below
        Some(ShAddOpcode::SH1ADD.global_opcode()),
        None,
        Some(RotateImmOpcode::RORI.global_opcode()),
        Some(CountZerosOpcode::CLZ.global_opcode()),
        None,
    ];
    for (i, (insn, expected_opcode)) in program.iter().zip(expected).enumerate() {
        let insn = insn.as_ref().unwrap();
        if let Some(op) = expected_opcode {
            assert_eq!(insn.opcode, op, "instruction {i}");
        } else {
            assert_ne!(
                insn.opcode,
                SystemOpcode::TERMINATE.global_opcode(),
                "instruction {i} unexpectedly unsupported"
            );
        }
    }
}

#[test]
fn bitmanip_program_executes_and_proves_with_rv64imb_config() {
    let words = golden_cases()
        .into_iter()
        .map(|(_, word, _, _)| word)
        .collect::<Vec<_>>();
    let mut instructions = rv64imb_transpiler().transpile(&words).unwrap();
    instructions.push(Some(Instruction {
        opcode: SystemOpcode::TERMINATE.global_opcode(),
        ..Default::default()
    }));
    let exe = VmExe::new(Program::new_without_debug_infos_with_option(
        &instructions,
        0,
    ));

    let config = Rv64ImBConfig::default();
    let executor = VmExecutor::new(config.clone()).unwrap();
    let instance = executor.instance(&exe).unwrap();
    instance.execute(vec![]).unwrap();

    air_test(Rv64ImBBuilder, config, exe);
}
