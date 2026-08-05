use std::{fs::read_dir, path::PathBuf};

use eyre::Result;
#[cfg(feature = "rvr")]
use openvm_circuit::arch::testing::assert_vm_states_equivalent;
use openvm_circuit::arch::{instructions::exe::VmExe, VmExecutor};
use openvm_riscv_circuit::RiscvImConfig;
use openvm_riscv_transpiler::{
    RiscvITranspilerExtension, RiscvIoTranspilerExtension, RiscvMTranspilerExtension,
};
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use openvm_toolchain_tests::decode_elf;
use openvm_transpiler::{transpiler::Transpiler, FromElf};

type F = BabyBear;

#[test]
#[ignore = "must run makefile"]
fn test_riscv_vector_runtime() -> Result<()> {
    let skip_list = ["rv64ui-p-ma_data", "rv64ui-p-fence_i"];
    let config = RiscvImConfig::default();
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("riscv-test-vectors/tests");
    let mut failures = Vec::new();
    for entry in read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().unwrap_or_default() == "" {
            let file_name = path.file_name().unwrap().to_str().unwrap();
            if skip_list.contains(&file_name) {
                continue;
            }
            println!("Running: {file_name}");
            let result = std::panic::catch_unwind(|| -> Result<_> {
                let elf = decode_elf(&path)?;
                let exe = VmExe::from_elf(
                    elf,
                    Transpiler::<F>::default()
                        .with_extension(RiscvITranspilerExtension)
                        .with_extension(RiscvMTranspilerExtension)
                        .with_extension(RiscvIoTranspilerExtension),
                )?;
                let executor = VmExecutor::new(config.clone())?;
                let instance = executor.instance(&exe)?;
                #[allow(unused_variables)]
                let state = instance.execute(vec![])?;

                #[cfg(feature = "rvr")]
                {
                    let interpreter_instance = executor.interpreter_instance(&exe)?;
                    let naive_state = interpreter_instance.execute(vec![])?;
                    assert_vm_states_equivalent(&state, &naive_state);
                }

                Ok(())
            });

            match result {
                Ok(Ok(_)) => println!("Passed!: {file_name}"),
                Ok(Err(e)) => {
                    println!("Failed: {file_name} with error: {e}");
                    failures.push(format!("{file_name}: {e:#}"));
                }
                Err(_) => {
                    println!("Panic occurred while running: {file_name}");
                    failures.push(format!("{file_name}: panicked"));
                }
            }
        }
    }

    if !failures.is_empty() {
        eyre::bail!("RISC-V runtime vectors failed:\n{}", failures.join("\n"));
    }
    Ok(())
}

// Running Prove tests only when CUDA is enabled because it is slow on CPU
#[test]
#[ignore = "long prover tests"]
fn test_riscv_vector_prove() -> Result<()> {
    use openvm_circuit::utils::air_test;
    use openvm_riscv_circuit::RiscvImBuilder;

    let config = RiscvImConfig::default();
    let skip_list = ["rv64ui-p-ma_data", "rv64ui-p-fence_i"];
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("riscv-test-vectors/tests");
    let mut failures = Vec::new();
    for entry in read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().unwrap_or_default() == "" {
            let file_name = path.file_name().unwrap().to_str().unwrap();
            if skip_list.contains(&file_name) {
                continue;
            }
            println!("Running: {file_name}");
            let elf = decode_elf(&path)?;
            let exe = VmExe::from_elf(
                elf,
                Transpiler::<F>::default()
                    .with_extension(RiscvITranspilerExtension)
                    .with_extension(RiscvMTranspilerExtension)
                    .with_extension(RiscvIoTranspilerExtension),
            )?;

            let result = std::panic::catch_unwind(|| {
                air_test(RiscvImBuilder, config.clone(), exe);
            });

            match result {
                Ok(_) => println!("Passed!: {file_name}"),
                Err(_) => {
                    println!("Panic occurred while running: {file_name}");
                    failures.push(format!("{file_name}: panicked"));
                }
            }
        }
    }

    if !failures.is_empty() {
        eyre::bail!("RISC-V proving vectors failed:\n{}", failures.join("\n"));
    }
    Ok(())
}
