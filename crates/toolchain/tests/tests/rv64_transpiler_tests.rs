use std::{
    fs::read,
    path::{Path, PathBuf},
};

use eyre::Result;
use openvm_instructions::{
    riscv::{RV64_IMM_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode, SystemOpcode,
};
use openvm_platform::memory::MEM_SIZE;
use openvm_riscv_transpiler::{
    BaseAluImmOpcode, BitwiseImmOpcode, LessThanImmOpcode, Rv64HintStoreOpcode,
    Rv64ITranspilerExtension, Rv64IoTranspilerExtension, Rv64MTranspilerExtension, Rv64Phantom,
    ShiftImmOpcode,
};
use openvm_stark_sdk::{openvm_stark_backend::p3_field::PrimeField32, p3_baby_bear::BabyBear};
use openvm_transpiler::{elf::Elf, transpiler::Transpiler};
use test_case::test_case;

type F = BabyBear;

fn get_elf(elf_path: impl AsRef<Path>) -> Result<Elf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data = read(dir.join(elf_path))?;
    let elf = Elf::decode(&data, MEM_SIZE as u32)?;
    Ok(elf)
}

fn rv64_transpiler() -> Transpiler<F> {
    Transpiler::<F>::default()
        .with_extension(Rv64ITranspilerExtension)
        .with_extension(Rv64MTranspilerExtension)
        .with_extension(Rv64IoTranspilerExtension)
}

fn encode_op_imm(funct3: u32, immediate: u32) -> u32 {
    const OPCODE_OP_IMM: u32 = 0b0010011;
    const RD: u32 = 3;
    const RS1: u32 = 5;

    ((immediate & 0xfff) << 20) | (RS1 << 15) | (funct3 << 12) | (RD << 7) | OPCODE_OP_IMM
}

#[test_case(-2048, 0xff_f800; "minimum")]
#[test_case(-1, 0xff_ffff; "negative_one")]
#[test_case(0, 0; "zero")]
#[test_case(2047, 0x7ff; "maximum")]
fn test_transpile_addi_immediate_boundaries(imm: i32, expected_c: u32) -> Result<()> {
    const OPCODE_OP_IMM: u32 = 0b0010011;
    const RD: usize = 3;
    const RS1: usize = 5;

    let encoded =
        (((imm as u32) & 0xfff) << 20) | ((RS1 as u32) << 15) | ((RD as u32) << 7) | OPCODE_OP_IMM;
    let program = rv64_transpiler().transpile(&[encoded])?;
    let instruction = program[0].as_ref().expect("ADDI should be emitted");

    assert_eq!(instruction.opcode, BaseAluImmOpcode::ADDI.global_opcode());
    assert_eq!(
        instruction.a.as_canonical_u32(),
        (RD * RV64_REGISTER_NUM_LIMBS) as u32
    );
    assert_eq!(
        instruction.b.as_canonical_u32(),
        (RS1 * RV64_REGISTER_NUM_LIMBS) as u32
    );
    assert_eq!(instruction.c.as_canonical_u32(), expected_c);
    assert_eq!(instruction.d.as_canonical_u32(), RV64_REGISTER_AS);
    assert_eq!(instruction.e.as_canonical_u32(), RV64_IMM_AS);

    Ok(())
}

#[test]
fn test_transpile_rv64_undecodable_word_as_unimp() -> Result<()> {
    const ADDI_X3_X0_1: u32 = 0x0010_0193;
    const UNDECODABLE_WORD: u32 = 0x1000_0200;

    let words = [ADDI_X3_X0_1, UNDECODABLE_WORD, ADDI_X3_X0_1];
    let program = rv64_transpiler().transpile(&words)?;

    assert_eq!(program.len(), words.len());
    assert_eq!(
        program[0].as_ref().unwrap().opcode,
        BaseAluImmOpcode::ADDI.global_opcode()
    );
    assert_eq!(
        program[2].as_ref().unwrap().opcode,
        BaseAluImmOpcode::ADDI.global_opcode()
    );
    let unimp = program[1].as_ref().expect("UNIMP should be emitted");
    assert_eq!(unimp.opcode, SystemOpcode::TERMINATE.global_opcode());
    assert_eq!(unimp.c.as_canonical_u32(), 2);

    Ok(())
}

#[test_case(0b100, 0x800, BitwiseImmOpcode::XORI.global_opcode().as_usize(), 0xff_f800; "xori")]
#[test_case(0b110, 0x7ff, BitwiseImmOpcode::ORI.global_opcode().as_usize(), 0x7ff; "ori")]
#[test_case(0b111, 0xfff, BitwiseImmOpcode::ANDI.global_opcode().as_usize(), 0xff_ffff; "andi")]
#[test_case(0b010, 0x800, LessThanImmOpcode::SLTI.global_opcode().as_usize(), 0xff_f800; "slti")]
#[test_case(0b011, 0x7ff, LessThanImmOpcode::SLTIU.global_opcode().as_usize(), 0x7ff; "sltiu")]
#[test_case(0b001, 63, ShiftImmOpcode::SLLI.global_opcode().as_usize(), 63; "slli")]
#[test_case(0b101, 63, ShiftImmOpcode::SRLI.global_opcode().as_usize(), 63; "srli")]
#[test_case(0b101, (0b0100000 << 5) | 63, ShiftImmOpcode::SRAI.global_opcode().as_usize(), 63; "srai")]
fn test_transpile_split_immediate_opcodes(
    funct3: u32,
    immediate: u32,
    expected_opcode: usize,
    expected_c: u32,
) -> Result<()> {
    let program = rv64_transpiler().transpile(&[encode_op_imm(funct3, immediate)])?;
    let instruction = program[0]
        .as_ref()
        .expect("immediate instruction should be emitted");

    assert_eq!(instruction.opcode.as_usize(), expected_opcode);
    assert_eq!(instruction.c.as_canonical_u32(), expected_c);
    assert_eq!(instruction.d.as_canonical_u32(), RV64_REGISTER_AS);
    assert_eq!(instruction.e.as_canonical_u32(), RV64_IMM_AS);

    Ok(())
}

// To create ELF directly from .S file, `brew install riscv-gnu-toolchain` and run
// `riscv64-unknown-elf-gcc -march=rv64im -mabi=lp64 -nostartfiles -e _start -Ttext 0x00200800
// -Wl,-N <name>.S -o <name>-from-as`
#[test_case("tests/data/rv64im-stress")]
#[test_case("tests/data/rv64im-intrin")]
fn test_decode_rv64_elf(elf_path: &str) -> Result<()> {
    let elf = get_elf(elf_path)?;
    assert!(
        !elf.instructions.is_empty(),
        "ELF should contain instructions"
    );
    Ok(())
}

#[test_case("tests/data/rv64im-stress")]
#[test_case("tests/data/rv64im-intrin")]
fn test_transpile_rv64_program(elf_path: &str) -> Result<()> {
    let elf = get_elf(elf_path)?;
    let program = rv64_transpiler().transpile(&elf.instructions)?;
    let non_none_count = program.iter().filter(|i| i.is_some()).count();
    assert!(
        non_none_count > 0,
        "Transpiled program should contain at least one instruction"
    );
    Ok(())
}

/// Verify that no instructions are transpiled as UNIMP.
/// UNIMP is transpiled as TERMINATE with exit code 2 (c = F::TWO).
/// Legitimate TERMINATE instructions (exit code 0 or 1) are expected and allowed.
#[test_case("tests/data/rv64im-stress")]
fn test_transpile_rv64_no_unimp(elf_path: &str) -> Result<()> {
    let elf = get_elf(elf_path)?;
    let program = rv64_transpiler().transpile(&elf.instructions)?;
    let terminate_opcode = SystemOpcode::TERMINATE.global_opcode();
    for (i, inst) in program.iter().enumerate() {
        if let Some(inst) = inst {
            if inst.opcode == terminate_opcode {
                assert_ne!(
                    inst.c.as_canonical_u32(),
                    2,
                    "Instruction at index {i} was transpiled as UNIMP: {inst:?}"
                );
            }
        }
    }
    Ok(())
}

/// Verify that the intrinsic ELF containing custom OpenVM opcodes
/// (TERMINATE, PHANTOM, HINT_STORED, HINT_BUFFER) transpiles correctly
/// and that the expected opcodes appear in the output.
#[test]
fn test_transpile_rv64_custom_opcodes() -> Result<()> {
    let elf = get_elf("tests/data/rv64im-intrin")?;
    let program = rv64_transpiler().transpile(&elf.instructions)?;

    let terminate_opcode = SystemOpcode::TERMINATE.global_opcode();
    let phantom_opcode = SystemOpcode::PHANTOM.global_opcode();
    let hint_stored_opcode = Rv64HintStoreOpcode::HINT_STORED.global_opcode();
    let hint_buffer_opcode = Rv64HintStoreOpcode::HINT_BUFFER.global_opcode();

    let mut found_terminate = false;
    let mut found_phantom = false;
    let mut found_hint_stored = false;
    let mut found_hint_buffer = false;

    for inst in program.iter().flatten() {
        if inst.opcode == terminate_opcode {
            found_terminate = true;
        } else if inst.opcode == phantom_opcode {
            found_phantom = true;
        } else if inst.opcode == hint_stored_opcode {
            found_hint_stored = true;
        } else if inst.opcode == hint_buffer_opcode {
            found_hint_buffer = true;
        }
    }

    assert!(
        found_terminate,
        "Expected TERMINATE opcode in transpiled program"
    );
    assert!(
        found_phantom,
        "Expected PHANTOM opcode in transpiled program"
    );
    assert!(
        found_hint_stored,
        "Expected HINT_STORED opcode in transpiled program"
    );
    assert!(
        found_hint_buffer,
        "Expected HINT_BUFFER opcode in transpiled program"
    );

    Ok(())
}

/// Verify that PHANTOM instructions carry the correct discriminant values
/// for PrintStr and HintInput.
#[test]
fn test_transpile_rv64_phantom_discriminants() -> Result<()> {
    let elf = get_elf("tests/data/rv64im-intrin")?;
    let program = rv64_transpiler().transpile(&elf.instructions)?;

    let phantom_opcode = SystemOpcode::PHANTOM.global_opcode();
    let hint_input_disc = Rv64Phantom::HintInput as u16;
    let print_str_disc = Rv64Phantom::PrintStr as u16;

    let mut found_hint_input = false;
    let mut found_print_str = false;

    for inst in program.iter().flatten() {
        if inst.opcode == phantom_opcode {
            // The discriminant is stored in the lower 16 bits of operand c
            let disc = inst.c.as_canonical_u32() as u16;
            if disc == hint_input_disc {
                found_hint_input = true;
            } else if disc == print_str_disc {
                found_print_str = true;
            }
        }
    }

    assert!(
        found_hint_input,
        "Expected PHANTOM(HintInput) with discriminant {hint_input_disc:#x}"
    );
    assert!(
        found_print_str,
        "Expected PHANTOM(PrintStr) with discriminant {print_str_disc:#x}"
    );

    Ok(())
}
