#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use eyre::Result;
    use openvm_circuit::{
        arch::{hasher::poseidon2::vm_poseidon2_hasher, ExecutionError, Streams, VmExecutor},
        system::memory::tree::public_values::UserPublicValuesProof,
        utils::{air_test, air_test_with_min_segments},
    };
    use openvm_instructions::exe::VmExe;
    use openvm_rv32im_circuit::{Rv32IConfig, Rv32ImConfig};
    use openvm_rv32im_guest::hint_load_by_key_encode;
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
        let config = Rv32IConfig::default();
        let elf = build_example_program_at_path(get_programs_dir!(), example_name, &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;
        air_test_with_min_segments(config, exe, vec![], min_segments);
        Ok(())
    }

    #[test_case("collatz", 1)]
    fn test_rv32im(example_name: &str, min_segments: usize) -> Result<()> {
        let config = Rv32ImConfig::default();
        let elf = build_example_program_at_path(get_programs_dir!(), example_name, &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(Rv32MTranspilerExtension),
        )?;
        air_test_with_min_segments(config, exe, vec![], min_segments);
        Ok(())
    }

    // #[test_case("fibonacci", 1)]
    #[test_case("collatz", 1)]
    fn test_rv32im_std(example_name: &str, min_segments: usize) -> Result<()> {
        let config = Rv32ImConfig::default();
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!(),
            example_name,
            ["std"],
            &config,
        )?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(Rv32MTranspilerExtension),
        )?;
        air_test_with_min_segments(config, exe, vec![], min_segments);
        Ok(())
    }

    #[test]
    fn test_read_vec() -> Result<()> {
        let config = Rv32IConfig::default();
        let elf = build_example_program_at_path(get_programs_dir!(), "hint", &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;
        let input = vec![[0, 1, 2, 3].map(F::from_canonical_u8).to_vec()];
        air_test_with_min_segments(config, exe, input, 1);
        Ok(())
    }

    #[test]
    fn test_hint_load_by_key() -> Result<()> {
        let config = Rv32IConfig::default();
        let elf = build_example_program_at_path(get_programs_dir!(), "hint_load_by_key", &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;
        // stdin will be read after reading kv_store
        let stdin = vec![[0, 1, 2].map(F::from_canonical_u8).to_vec()];
        let mut streams: Streams<F> = stdin.into();
        let input = vec![[0, 1, 2, 3].map(F::from_canonical_u8).to_vec()];
        streams.kv_store = Arc::new(HashMap::from([(
            "key".as_bytes().to_vec(),
            hint_load_by_key_encode(&input),
        )]));
        air_test_with_min_segments(config, exe, streams, 1);
        Ok(())
    }

    #[test]
    fn test_read() -> Result<()> {
        let config = Rv32IConfig::default();
        let elf = build_example_program_at_path(get_programs_dir!(), "read", &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;

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
        let config = Rv32IConfig::default();
        let elf = build_example_program_at_path(get_programs_dir!(), "reveal", &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;
        let executor = VmExecutor::<F, _>::new(config.clone());
        let final_memory = executor.execute(exe, vec![])?.unwrap();
        let hasher = vm_poseidon2_hasher();
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
        let config = Rv32IConfig::default();
        let elf = build_example_program_at_path(get_programs_dir!(), "print", &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;
        air_test(config, exe);
        Ok(())
    }

    #[test]
    fn test_heap_overflow() -> Result<()> {
        let config = Rv32ImConfig::default();
        let elf = build_example_program_at_path(get_programs_dir!(), "heap_overflow", &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;

        let executor = VmExecutor::<F, _>::new(config.clone());
        match executor.execute(exe, vec![[0, 0, 0, 1].map(F::from_canonical_u8).to_vec()]) {
            Err(ExecutionError::FailedWithExitCode(_)) => Ok(()),
            Err(_) => panic!("should fail with `FailedWithExitCode`"),
            Ok(_) => panic!("should fail"),
        }
    }

    #[test]
    fn test_hashmap() -> Result<()> {
        let config = Rv32ImConfig::default();
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!(),
            "hashmap",
            ["std"],
            &config,
        )?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;
        air_test(config, exe);
        Ok(())
    }

    #[test]
    fn test_tiny_mem_test() -> Result<()> {
        let config = Rv32ImConfig::default();
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!(),
            "tiny-mem-test",
            ["heap-embedded-alloc"],
            &config,
        )?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;
        air_test(config, exe);
        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_load_x0() {
        let config = Rv32ImConfig::default();
        let elf = build_example_program_at_path(get_programs_dir!(), "load_x0", &config).unwrap();
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )
        .unwrap();
        let executor = VmExecutor::<F, _>::new(config.clone());
        executor.execute(exe, vec![]).unwrap();
    }

    #[test_case("getrandom", vec!["getrandom", "getrandom-unsupported"])]
    #[test_case("getrandom", vec!["getrandom"])]
    #[test_case("getrandom_v02", vec!["getrandom-v02", "getrandom-unsupported"])]
    #[test_case("getrandom_v02", vec!["getrandom-v02/custom"])]
    fn test_getrandom_unsupported(program: &str, features: Vec<&str>) {
        let config = Rv32ImConfig::default();
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!(),
            program,
            &features,
            &config,
        )
        .unwrap();
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )
        .unwrap();
        air_test(config, exe);
    }
}
