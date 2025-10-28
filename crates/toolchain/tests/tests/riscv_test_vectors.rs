use std::{fs::read_dir, path::PathBuf};

use eyre::Result;
use openvm_circuit::arch::{instructions::exe::VmExe, VmExecutor};
use openvm_rv32im_circuit::Rv32ImConfig;
use openvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use openvm_toolchain_tests::decode_elf;
use openvm_transpiler::{transpiler::Transpiler, FromElf};

type F = BabyBear;

#[test]
#[ignore = "must run makefile"]
fn test_rv32im_riscv_vector_runtime() -> Result<()> {
    let skip_list = ["rv32ui-p-ma_data", "rv32ui-p-fence_i"];
    let config = Rv32ImConfig::default();
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
            let result = std::panic::catch_unwind(|| -> Result<_> {
                let elf = decode_elf(&path)?;
                let exe = VmExe::from_elf(
                    elf,
                    Transpiler::<F>::default()
                        .with_extension(Rv32ITranspilerExtension)
                        .with_extension(Rv32MTranspilerExtension)
                        .with_extension(Rv32IoTranspilerExtension),
                )?;
                let executor = VmExecutor::new(config.clone())?;
                let interpreter = executor.instance(&exe)?;
                let _state = interpreter.execute(vec![], None)?;
                Ok(())
            });

            match result {
                Ok(Ok(_)) => println!("Passed!: {}", file_name),
                Ok(Err(e)) => println!("Failed: {} with error: {}", file_name, e),
                Err(_) => panic!("Panic occurred while running: {}", file_name),
            }
        }
    }

    Ok(())
}

// Running Prove tests only when CUDA is enabled because it is slow on CPU
#[test]
#[ignore = "long prover tests"]
fn test_rv32im_riscv_vector_prove() -> Result<()> {
    use openvm_circuit::utils::air_test;
    use openvm_rv32im_circuit::Rv32ImBuilder;

    let config = Rv32ImConfig::default();
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
            let elf = decode_elf(&path)?;
            let exe = VmExe::from_elf(
                elf,
                Transpiler::<F>::default()
                    .with_extension(Rv32ITranspilerExtension)
                    .with_extension(Rv32MTranspilerExtension)
                    .with_extension(Rv32IoTranspilerExtension),
            )?;

            let result = std::panic::catch_unwind(|| {
                air_test(Rv32ImBuilder, config.clone(), exe);
            });

            match result {
                Ok(_) => println!("Passed!: {}", file_name),
                Err(_) => println!("Panic occurred while running: {}", file_name),
            }
        }
    }

    Ok(())
}
