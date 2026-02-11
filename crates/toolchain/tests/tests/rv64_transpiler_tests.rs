use std::{
    fs::read,
    path::{Path, PathBuf},
};

use eyre::Result;
use openvm_circuit::arch::{instructions::exe::VmExe, VmExecutor};
use openvm_instructions::{riscv::RV32_MEMORY_AS, LocalOpcode, SystemOpcode};
use openvm_platform::memory::MEM_SIZE;
use openvm_rv64im_circuit::Rv64ImConfig;
use openvm_rv64im_transpiler::{
    Rv64HintStoreOpcode, Rv64ITranspilerExtension, Rv64IoTranspilerExtension,
    Rv64MTranspilerExtension, Rv64Phantom,
};
use openvm_stark_sdk::openvm_stark_backend::p3_field::{FieldAlgebra, PrimeField32};
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
    execute_rv64_elf_with_input(elf_path, vec![])
}

fn execute_rv64_elf_with_input(
    elf_path: &str,
    input: Vec<Vec<F>>,
) -> Result<openvm_circuit::arch::VmState<F, openvm_circuit::system::memory::online::GuestMemory>> {
    let elf = get_elf(elf_path)?;
    let exe = VmExe::from_elf(elf, rv64_transpiler())?;
    let config = Rv64ImConfig::default();
    let executor = VmExecutor::new(config)?;
    let instance = executor.instance(&exe)?;
    Ok(instance.execute(input, None)?)
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
    assert!(
        result.is_err(),
        "expected execution to fail with non-zero exit code"
    );
}

/// Execute the intrinsic test through the full pipeline:
/// ELF decode -> transpile -> interpreter execution.
/// The intrin test exercises PHANTOM (HintInput, PrintStr), HINT_STORED, HINT_BUFFER,
/// and TERMINATE. We provide distinguishable hint data so we can verify the values
/// written to memory by hint_stored and hint_buffer.
///
/// Data flow:
///   - phantom_hint_input: pops the 72-element input Vec<F>, writes 4-byte length prefix
///     + data to hint_stream → [F(72), F(0), F(0), F(0), F(10), F(11), ..., F(81)]
///   - hint_stored (1 dword at 0x300100): pops 8 elements → [72, 0, 0, 0, 10, 11, 12, 13]
///   - hint_buffer (8 dwords at 0x300200): pops 64 elements → sequential bytes 14..77
#[test]
fn test_execute_rv64_intrin() -> Result<()> {
    // Use distinguishable input values so we can verify hint_stored and hint_buffer
    // wrote the correct data to memory.
    let hint_data: Vec<F> = (0u32..72)
        .map(|i| F::from_canonical_u32(i + 10))
        .collect();
    let state = execute_rv64_elf_with_input("tests/data/rv64im-intrin", vec![hint_data])?;

    // Verify hint_stored wrote the correct dword at 0x300100:
    // first 4 bytes are the length prefix (72 as u32 LE), next 4 are data[0..4].
    let stored: [u8; 8] =
        unsafe { state.memory.read::<u8, 8>(RV32_MEMORY_AS, 0x300100) };
    assert_eq!(
        stored,
        [72, 0, 0, 0, 10, 11, 12, 13],
        "hint_stored: expected length prefix (72,0,0,0) + first 4 data bytes (10,11,12,13)"
    );

    // Verify hint_buffer wrote 8 correct dwords starting at 0x300200.
    for dword_idx in 0u32..8 {
        let addr = 0x300200 + dword_idx * 8;
        let actual: [u8; 8] =
            unsafe { state.memory.read::<u8, 8>(RV32_MEMORY_AS, addr) };
        let expected: [u8; 8] =
            std::array::from_fn(|j| (14 + dword_idx * 8 + j as u32) as u8);
        assert_eq!(
            actual, expected,
            "hint_buffer dword {dword_idx} at {addr:#x}"
        );
    }

    Ok(())
}
