use std::{
    fs::read,
    path::{Path, PathBuf},
};

use eyre::Result;
use openvm_circuit::arch::{instructions::exe::VmExe, VmExecutor};
use openvm_instructions::{LocalOpcode, SystemOpcode};
use openvm_platform::memory::MEM_SIZE;
use openvm_rv64im_circuit::Rv64ImConfig;
use openvm_rv64im_transpiler::{
    Rv64HintStoreOpcode, Rv64ITranspilerExtension, Rv64IoTranspilerExtension,
    Rv64MTranspilerExtension, Rv64Phantom,
};
use openvm_stark_sdk::openvm_stark_backend::p3_field::PrimeField32;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use openvm_transpiler::{elf::Elf, transpiler::Transpiler, FromElf};
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

// To create ELF directly from .S file, `brew install riscv-gnu-toolchain` and run
// `riscv64-unknown-elf-gcc -march=rv64im -mabi=lp64 -nostartfiles -e _start -Ttext 0x00200800 -Wl,-N <name>.S -o <name>-from-as`
#[test_case("tests/data/rv64im-stress")]
#[test_case("tests/data/rv64im-intrin")]
fn test_decode_rv64_elf(elf_path: &str) -> Result<()> {
    let elf = get_elf(elf_path)?;
    assert_eq!(format!("{:?}", elf.class), "ELF64");
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

// ---------------------------------------------------------------------------
// End-to-end execution tests
// ---------------------------------------------------------------------------

fn execute_rv64_elf(elf_path: &str) -> Result<openvm_circuit::arch::VmState<F, openvm_circuit::system::memory::online::GuestMemory>> {
    let elf = get_elf(elf_path)?;
    let exe = VmExe::from_elf(elf, rv64_transpiler())?;
    let config = Rv64ImConfig::default();
    let executor = VmExecutor::new(config)?;
    let instance = executor.instance(&exe)?;
    Ok(instance.execute(vec![] as Vec<Vec<F>>, None)?)
}

/// Execute the comprehensive RV64IM stress test through the full pipeline:
/// ELF decode -> transpile -> interpreter execution.
/// The stress test covers every RV64IM instruction and terminates with exit code 0 on success.
#[test]
fn test_execute_rv64_stress() -> Result<()> {
    let _state = execute_rv64_elf("tests/data/rv64im-stress")?;
    Ok(())
}

/// Verify that a program with a deliberate assertion failure (assert_eq(0, 1))
/// is detected as a non-zero exit code error. This ensures the test harness
/// actually catches failures in guest programs.
#[test]
fn test_execute_rv64_fail_detected() {
    let result = execute_rv64_elf("tests/data/rv64im-fail");
    assert!(result.is_err(), "expected execution to fail with non-zero exit code");
}
