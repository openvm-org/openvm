use std::{
    fs::{read, read_dir},
    path::PathBuf,
};

use ax_stark_sdk::config::setup_tracing;
use axvm_circuit::{
    arch::{VirtualMachine, VmConfig},
    sdk::{air_test, air_test_with_min_segments},
};
use axvm_platform::memory::MEM_SIZE;
use color_eyre::eyre::Result;
use p3_baby_bear::BabyBear;
use test_case::test_case;

use crate::{elf::Elf, rrs::transpile, AxVmExe};

type F = BabyBear;

fn setup_vm_from_elf(elf_path: &str, config: VmConfig) -> Result<(VirtualMachine<F>, AxVmExe<F>)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data = read(dir.join(elf_path))?;
    let elf = Elf::decode(&data, MEM_SIZE as u32)?;
    let vm = VirtualMachine::new(config);
    Ok((vm, elf.into()))
}

#[test]
fn test_decode_elf() -> Result<()> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data = read(dir.join("data/rv32im-empty-program-elf"))?;
    let elf = Elf::decode(&data, MEM_SIZE as u32)?;
    dbg!(elf);
    Ok(())
}

// To create ELF directly from .S file, `brew install riscv-gnu-toolchain` and run
// `riscv64-unknown-elf-gcc -march=rv32im -mabi=ilp32 -nostartfiles -e _start -Ttext 0 fib.S -o rv32im-fib-from-as`
// riscv64-unknown-elf-gcc supports rv32im if you set -march target
#[test_case("data/rv32im-fib-from-as")]
#[test_case("data/rv32im-intrin-from-as")]
fn test_generate_program(elf_path: &str) -> Result<()> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data = read(dir.join(elf_path))?;
    let elf = Elf::decode(&data, MEM_SIZE as u32)?;
    let program = transpile::<BabyBear>(&elf.instructions);
    for instruction in program {
        println!("{:?}", instruction);
    }
    Ok(())
}

#[test_case("data/rv32im-fibonacci-program-elf-release")]
#[test_case("data/rv32im-exp-from-as")]
#[test_case("data/rv32im-fib-from-as")]
fn test_rv32im_runtime(elf_path: &str) -> Result<()> {
    setup_tracing();
    let config = VmConfig::rv32im();
    let (vm, exe) = setup_vm_from_elf(elf_path, config)?;
    vm.execute(exe, vec![])?;
    Ok(())
}

#[test_case("data/rv32im-fibonacci-program-elf-release")]
fn test_rv32i_prove(elf_path: &str) -> Result<()> {
    let config = VmConfig::rv32i();
    let (vm, exe) = setup_vm_from_elf(elf_path, config)?;
    air_test(vm, exe);
    Ok(())
}

#[test_case("data/rv32im-fibonacci-large-program-elf-release")]
fn test_rv32i_continuations(elf_path: &str) -> Result<()> {
    let config = VmConfig {
        max_segment_len: (1 << 18) - 1,
        ..VmConfig::rv32i()
    };
    let (vm, exe) = setup_vm_from_elf(elf_path, config)?;
    air_test_with_min_segments(vm, exe, vec![], 3);
    Ok(())
}

#[test_case("data/rv32im-intrin-from-as")]
fn test_intrinsic_runtime(elf_path: &str) -> Result<()> {
    setup_tracing();
    let config = VmConfig::rv32im().add_canonical_modulus();
    let (vm, exe) = setup_vm_from_elf(elf_path, config)?;
    vm.execute(exe, vec![])?;
    Ok(())
}

#[test]
fn test_terminate_runtime() -> Result<()> {
    setup_tracing();
    let config = VmConfig::rv32i();
    let (vm, exe) = setup_vm_from_elf("data/rv32im-terminate-from-as", config)?;
    air_test(vm, exe);
    Ok(())
}

#[test]
fn test_rv32im_riscv_vector_runtime() -> Result<()> {
    let skip_list = ["rv32i-p-ma_data", "rv32i-p-fence_i"];
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("rv32im-test-vectors/tests");
    for entry in read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().unwrap_or_default() == "" {
            let file_name = path.file_name().unwrap().to_str().unwrap();
            if skip_list.contains(&file_name) {
                continue;
            }
            println!("Running: {}", file_name);
            let result = std::panic::catch_unwind(|| test_rv32im_runtime(path.to_str().unwrap()));

            match result {
                Ok(Ok(_)) => println!("Passed!: {}", file_name),
                Ok(Err(e)) => println!("Failed: {} with error: {}", file_name, e),
                Err(_) => panic!("Panic occurred while running: {}", file_name),
            }
        }
    }

    Ok(())
}

#[test]
fn test_rv32im_riscv_vector_prove() -> Result<()> {
    let skip_list = [
        "rv32i-p-ma_data",
        "rv32i-p-fence_i",
        "rv32i-p-lui",
        "rv32i-p-sra",
    ];
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("rv32im-test-vectors/tests");
    for entry in read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().unwrap_or_default() == "" {
            let file_name = path.file_name().unwrap().to_str().unwrap();
            if skip_list.contains(&file_name) {
                continue;
            }
            println!("Running: {}", file_name);
            let result = std::panic::catch_unwind(|| test_rv32i_prove(path.to_str().unwrap()));

            match result {
                Ok(Ok(_)) => println!("Passed!: {}", file_name),
                Ok(Err(e)) => println!("Failed: {} with error: {}", file_name, e),
                Err(_) => panic!("Panic occurred while running: {}", file_name),
            }
        }
    }

    Ok(())
}
