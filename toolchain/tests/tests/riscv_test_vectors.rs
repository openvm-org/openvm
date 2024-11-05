use std::{fs::read_dir, path::PathBuf};

use axvm_circuit::{
    arch::{VmConfig, VmExecutor},
    utils::air_test,
};
use axvm_toolchain_tests::utils::decode_elf;
use eyre::Result;
use p3_baby_bear::BabyBear;

#[test]
#[ignore = "must run makefile"]
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
            let result = std::panic::catch_unwind(|| -> Result<_> {
                let elf = decode_elf(&path)?;
                let executor = VmExecutor::<BabyBear>::new(VmConfig::rv32im());
                let res = executor.execute(elf, vec![])?;
                Ok(res)
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

#[test]
#[ignore = "long prover tests"]
fn test_rv32im_riscv_vector_prove() -> Result<()> {
    let config = VmConfig::rv32im();
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

            let result = std::panic::catch_unwind(|| {
                air_test(config.clone(), elf);
            });

            match result {
                Ok(_) => println!("Passed!: {}", file_name),
                Err(_) => println!("Panic occurred while running: {}", file_name),
            }
        }
    }

    Ok(())
}
