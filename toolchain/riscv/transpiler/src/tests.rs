use std::{
    fs::{read, read_dir},
    path::{Path, PathBuf},
};

use ax_stark_sdk::config::setup_tracing;
use axvm_build::{build_guest_package, get_package, guest_methods, GuestOptions};
use axvm_circuit::{
    arch::{VmConfig, VmExecutor},
    sdk::{air_test, air_test_with_min_segments},
};
use axvm_platform::memory::MEM_SIZE;
use eyre::Result;
use p3_baby_bear::BabyBear;
use tempfile::tempdir;
use test_case::test_case;

use crate::{elf::Elf, rrs::transpile, AxVmExe};

type F = BabyBear;

fn setup_executor_from_elf(
    elf_path: impl AsRef<Path>,
    config: VmConfig,
) -> Result<(VmExecutor<F>, AxVmExe<F>)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data = read(dir.join(elf_path))?;
    let elf = Elf::decode(&data, MEM_SIZE as u32)?;
    let executor = VmExecutor::new(config);
    Ok((executor, elf.into()))
}

fn get_examples_dir() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    dir.push("examples");
    dir
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

#[test_case("data/rv32im-exp-from-as")]
#[test_case("data/rv32im-fib-from-as")]
fn test_rv32im_runtime(elf_path: &str) -> Result<()> {
    setup_tracing();
    let config = VmConfig::rv32im();
    let (executor, exe) = setup_executor_from_elf(elf_path, config)?;
    executor.execute(exe, vec![])?;
    Ok(())
}

#[test_case("fibonacci/program", 1)]
#[test_case("fibonacci-large/program", 3)]
fn test_rv32i_prove(examples_path: &str, min_segments: usize) -> Result<()> {
    let pkg = get_package(get_examples_dir().join(examples_path));
    let target_dir = tempdir()?;
    let guest_opts = GuestOptions::default().into();
    build_guest_package(&pkg, &target_dir, &guest_opts, None);
    let elf_path = guest_methods(&pkg, &target_dir, &[]).pop().unwrap();
    let config = VmConfig {
        max_segment_len: (1 << 18) - 1,
        ..VmConfig::rv32i()
    };
    let (_, exe) = setup_executor_from_elf(elf_path, config.clone())?;
    air_test_with_min_segments(config, exe, vec![], min_segments);
    Ok(())
}

#[test_case("data/rv32im-intrin-from-as")]
fn test_intrinsic_runtime(elf_path: &str) -> Result<()> {
    setup_tracing();
    let config = VmConfig::rv32im().add_canonical_modulus();
    let (executor, exe) = setup_executor_from_elf(elf_path, config)?;
    executor.execute(exe, vec![])?;
    Ok(())
}

#[test]
fn test_terminate_runtime() -> Result<()> {
    setup_tracing();
    let config = VmConfig::rv32i();
    let (_, exe) = setup_executor_from_elf("data/rv32im-terminate-from-as", config.clone())?;
    air_test(config, exe);
    Ok(())
}

#[test]
fn test_rv32im_riscv_vector_runtime() -> Result<()> {
    let skip_list = ["rv32ui-p-ma_data", "rv32ui-p-fence_i"];
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
    let config = VmConfig {
        max_segment_len: (1 << 20) - 1,
        ..VmConfig::rv32im()
    };
    let skip_list = ["rv32ui-p-ma_data", "rv32ui-p-fence_i"];
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

            let (_, exe) = setup_executor_from_elf(path.to_str().unwrap(), config.clone())?;
            let result = std::panic::catch_unwind(|| {
                air_test(config.clone(), exe);
            });

            match result {
                Ok(_) => println!("Passed!: {}", file_name),
                Err(_) => println!("Panic occurred while running: {}", file_name),
            }
        }
    }

    Ok(())
}
