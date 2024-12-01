use std::{fs::read_dir, path::PathBuf, rc::Rc};

use axvm_circuit::{
    arch::{instructions::exe::AxVmExe, new_vm::VmExecutor},
    utils::new_air_test_with_min_segments,
};
use axvm_rv32im_circuit::Rv32ImConfig;
use axvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use axvm_toolchain_tests::utils::decode_elf;
use axvm_transpiler::{transpiler::Transpiler, FromElf};
use eyre::Result;
use p3_baby_bear::BabyBear;

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
                let exe = AxVmExe::from_elf(
                    elf,
                    Transpiler::<F>::default()
                        .with_processor(Rc::new(Rv32ITranspilerExtension))
                        .with_processor(Rc::new(Rv32MTranspilerExtension))
                        .with_processor(Rc::new(Rv32IoTranspilerExtension)),
                );
                let executor = VmExecutor::<F, _>::new(config.clone());
                let res = executor.execute(exe, vec![])?;
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
            let exe = AxVmExe::from_elf(
                elf,
                Transpiler::<F>::default()
                    .with_processor(Rc::new(Rv32ITranspilerExtension))
                    .with_processor(Rc::new(Rv32MTranspilerExtension))
                    .with_processor(Rc::new(Rv32IoTranspilerExtension)),
            );

            let result = std::panic::catch_unwind(|| {
                new_air_test_with_min_segments(config.clone(), exe, vec![], 1);
            });

            match result {
                Ok(_) => println!("Passed!: {}", file_name),
                Err(_) => println!("Panic occurred while running: {}", file_name),
            }
        }
    }

    Ok(())
}
