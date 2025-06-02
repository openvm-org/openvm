#[cfg(test)]
mod tests {
    use eyre::Result;
    use openvm_circuit::{
        arch::{hasher::poseidon2::vm_poseidon2_hasher, ExecutionError, VmExecutor},
        system::memory::tree::public_values::UserPublicValuesProof,
        utils::{air_test, air_test_with_min_segments},
    };
    use openvm_instructions::exe::VmExe;
    use openvm_rv32im_circuit::{Rv32IConfig, Rv32ImConfig};
    use openvm_rv32im_transpiler::{
        Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
    };
    use openvm_stark_sdk::{openvm_stark_backend::p3_field::FieldAlgebra, p3_baby_bear::BabyBear};
    use openvm_toolchain_tests::{
        build_example_program_at_path, build_example_program_at_path_with_features,
        get_programs_dir,
    };
    use openvm_transpiler::{transpiler::Transpiler, FromElf};
    use test_case::test_case;

    type F = BabyBear;

    #[test_case("fibonacci", 1)]
    fn test_rv32i(example_name: &str, min_segments: usize) -> Result<()> {
        let elf = build_example_program_at_path(get_programs_dir!(), example_name)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;
        let config = Rv32IConfig::default();
        air_test_with_min_segments(config, exe, vec![], min_segments);
        Ok(())
    }

    #[test_case("collatz", 1)]
    fn test_rv32im(example_name: &str, min_segments: usize) -> Result<()> {
        let elf = build_example_program_at_path(get_programs_dir!(), example_name)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(Rv32MTranspilerExtension),
        )?;
        let config = Rv32ImConfig::default();
        air_test_with_min_segments(config, exe, vec![], min_segments);
        Ok(())
    }

    // #[test_case("fibonacci", 1)]
    #[test_case("collatz", 1)]
    fn test_rv32im_std(example_name: &str, min_segments: usize) -> Result<()> {
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!(),
            example_name,
            ["std"],
        )?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(Rv32MTranspilerExtension),
        )?;
        let config = Rv32ImConfig::default();
        air_test_with_min_segments(config, exe, vec![], min_segments);
        Ok(())
    }

    #[test]
    fn test_read_vec() -> Result<()> {
        let elf = build_example_program_at_path(get_programs_dir!(), "hint")?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;
        let config = Rv32IConfig::default();
        let input = vec![[0, 1, 2, 3].map(F::from_canonical_u8).to_vec()];
        air_test_with_min_segments(config, exe, input, 1);
        Ok(())
    }

    #[test]
    fn test_read() -> Result<()> {
        let elf = build_example_program_at_path(get_programs_dir!(), "read")?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;
        let config = Rv32IConfig::default();

        #[derive(serde::Serialize)]
        struct Foo {
            bar: u32,
            baz: Vec<u32>,
        }
        let foo = Foo {
            bar: 42,
            baz: vec![0, 1, 2, 3],
        };
        let serialized_foo = openvm::serde::to_vec(&foo).unwrap();
        let input = serialized_foo
            .into_iter()
            .flat_map(|w| w.to_le_bytes())
            .map(F::from_canonical_u8)
            .collect();
        air_test_with_min_segments(config, exe, vec![input], 1);
        Ok(())
    }

    #[test]
    fn test_reveal() -> Result<()> {
        let elf = build_example_program_at_path(get_programs_dir!(), "reveal")?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;
        let config = Rv32IConfig::default();
        let executor = VmExecutor::<F, _>::new(config.clone());
        let final_memory = executor.execute(exe, vec![])?.unwrap();
        let hasher = vm_poseidon2_hasher::<F>();
        let pv_proof = UserPublicValuesProof::compute(
            config.system.memory_config.memory_dimensions(),
            64,
            &hasher,
            &final_memory,
        );
        let mut bytes = [0u8; 32];
        for (i, byte) in bytes.iter_mut().enumerate() {
            *byte = i as u8;
        }
        assert_eq!(
            pv_proof.public_values,
            bytes
                .into_iter()
                .chain(
                    [123, 0, 456, 0u32, 0u32, 0u32, 0u32, 0u32]
                        .into_iter()
                        .flat_map(|x| x.to_le_bytes())
                )
                .map(F::from_canonical_u8)
                .collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn test_print() -> Result<()> {
        let elf = build_example_program_at_path(get_programs_dir!(), "print")?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;
        let config = Rv32IConfig::default();
        air_test(config, exe);
        Ok(())
    }

    #[test]
    fn test_heap_overflow() -> Result<()> {
        let elf = build_example_program_at_path(get_programs_dir!(), "heap_overflow")?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;
        let config = Rv32ImConfig::default();

        let executor = VmExecutor::<F, _>::new(config.clone());
        match executor.execute(exe, vec![[0, 0, 0, 1].map(F::from_canonical_u8).to_vec()]) {
            Err(ExecutionError::FailedWithExitCode(_)) => Ok(()),
            Err(_) => panic!("should fail with `FailedWithExitCode`"),
            Ok(_) => panic!("should fail"),
        }
    }

    #[test]
    fn test_hashmap() -> Result<()> {
        let elf =
            build_example_program_at_path_with_features(get_programs_dir!(), "hashmap", ["std"])?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;
        let config = Rv32ImConfig::default();
        air_test(config, exe);
        Ok(())
    }

    #[test]
    fn test_tiny_mem_test() -> Result<()> {
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!(),
            "tiny-mem-test",
            ["heap-embedded-alloc"],
        )?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;
        let config = Rv32ImConfig::default();
        air_test(config, exe);
        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_load_x0() {
        let elf = build_example_program_at_path(get_programs_dir!(), "load_x0").unwrap();
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )
        .unwrap();
        let config = Rv32ImConfig::default();
        let executor = VmExecutor::<F, _>::new(config.clone());
        executor.execute(exe, vec![]).unwrap();
    }
}
