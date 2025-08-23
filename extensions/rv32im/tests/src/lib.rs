#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use eyre::Result;
    use openvm_circuit::{
        arch::{hasher::poseidon2::vm_poseidon2_hasher, ExecutionError, Streams, VmExecutor},
        system::memory::merkle::public_values::UserPublicValuesProof,
        utils::{air_test, air_test_with_min_segments, test_system_config},
    };
    use openvm_instructions::{exe::VmExe, instruction::Instruction, LocalOpcode, SystemOpcode};
    use openvm_rv32im_circuit::{Rv32IBuilder, Rv32IConfig, Rv32ImBuilder, Rv32ImConfig};
    use openvm_rv32im_guest::hint_load_by_key_encode;
    use openvm_rv32im_transpiler::{
        DivRemOpcode, MulHOpcode, MulOpcode, Rv32ITranspilerExtension, Rv32IoTranspilerExtension,
        Rv32MTranspilerExtension,
    };
    use openvm_stark_sdk::{openvm_stark_backend::p3_field::FieldAlgebra, p3_baby_bear::BabyBear};
    use openvm_toolchain_tests::{
        build_example_program_at_path, build_example_program_at_path_with_features,
        get_programs_dir,
    };
    use openvm_transpiler::{transpiler::Transpiler, FromElf};
    use strum::IntoEnumIterator;
    use test_case::test_case;

    type F = BabyBear;

    #[cfg(test)]
    fn test_rv32im_config() -> Rv32ImConfig {
        Rv32ImConfig {
            rv32i: Rv32IConfig {
                system: test_system_config(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test_case("fibonacci", 1)]
    fn test_rv32i(example_name: &str, min_segments: usize) -> Result<()> {
        let config = Rv32IConfig::default();
        let elf = build_example_program_at_path(get_programs_dir!(), example_name, &config)?;
        let mut exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;
        change_rv32m_insn_to_nop(&mut exe);
        air_test_with_min_segments(Rv32IBuilder, config, exe, vec![], min_segments);
        Ok(())
    }

    #[test_case("fibonacci", 1)]
    #[test_case("collatz", 1)]
    fn test_rv32im(example_name: &str, min_segments: usize) -> Result<()> {
        let config = test_rv32im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), example_name, &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(Rv32MTranspilerExtension),
        )?;
        air_test_with_min_segments(Rv32ImBuilder, config, exe, vec![], min_segments);
        Ok(())
    }

    #[test_case("fibonacci", 1)]
    #[test_case("collatz", 1)]
    fn test_rv32im_std(example_name: &str, min_segments: usize) -> Result<()> {
        let config = test_rv32im_config();
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
        air_test_with_min_segments(Rv32ImBuilder, config, exe, vec![], min_segments);
        Ok(())
    }

    #[test]
    fn test_read_vec() -> Result<()> {
        let config = test_rv32im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), "hint", &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;
        let input = vec![[0, 1, 2, 3].map(F::from_canonical_u8).to_vec()];
        air_test_with_min_segments(Rv32ImBuilder, config, exe, input, 1);
        Ok(())
    }

    #[test]
    fn test_hint_load_by_key() -> Result<()> {
        let config = test_rv32im_config();
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
        air_test_with_min_segments(Rv32ImBuilder, config, exe, streams, 1);
        Ok(())
    }

    #[test]
    fn test_read() -> Result<()> {
        let config = test_rv32im_config();
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
        air_test_with_min_segments(Rv32ImBuilder, config, exe, vec![input], 1);
        Ok(())
    }

    #[test]
    fn test_reveal() -> Result<()> {
        let config = test_rv32im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), "reveal", &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;

        let executor = VmExecutor::new(config.clone())?;
        let instance = executor.instance(&exe)?;
        let state = instance.execute(vec![], None)?;
        let final_memory = state.memory.memory;
        let hasher = vm_poseidon2_hasher::<F>();
        let pv_proof = UserPublicValuesProof::compute(
            config.as_ref().memory_config.memory_dimensions(),
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
        let config = test_rv32im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), "print", &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;
        air_test(Rv32ImBuilder, config, exe);
        Ok(())
    }

    #[test]
    fn test_heap_overflow() -> Result<()> {
        let config = test_rv32im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), "heap_overflow", &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;

        let executor = VmExecutor::new(config)?;
        let instance = executor.instance(&exe)?;
        let input = vec![[0, 0, 0, 1].map(F::from_canonical_u8).to_vec()];
        match instance.execute(input.clone(), None) {
            Err(ExecutionError::FailedWithExitCode(_)) => Ok(()),
            Err(_) => panic!("should fail with `FailedWithExitCode`"),
            Ok(_) => panic!("should fail"),
        }
    }

    #[test]
    fn test_hashmap() -> Result<()> {
        let config = test_rv32im_config();
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
        air_test(Rv32ImBuilder, config, exe);
        Ok(())
    }

    #[test]
    fn test_tiny_mem_test() -> Result<()> {
        let config = test_rv32im_config();
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
        air_test(Rv32ImBuilder, config, exe);
        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_load_x0() {
        let config = test_rv32im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), "load_x0", &config).unwrap();
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )
        .unwrap();
        let executor = VmExecutor::new(config).unwrap();
        let instance = executor.instance(&exe).unwrap();
        instance.execute(vec![], None).unwrap();
    }

    #[test_case("getrandom", vec!["getrandom", "getrandom-unsupported"])]
    #[test_case("getrandom", vec!["getrandom"])]
    #[test_case("getrandom_v02", vec!["getrandom-v02", "getrandom-unsupported"])]
    #[test_case("getrandom_v02", vec!["getrandom-v02/custom"])]
    fn test_getrandom_unsupported(program: &str, features: Vec<&str>) {
        let config = test_rv32im_config();
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
        air_test(Rv32ImBuilder, config, exe);
    }

    // For testing programs that should only execute RV32I:
    // The ELF might still have Mul instructions even though the program doesn't use them. We
    // mask those to NOP here.
    fn change_rv32m_insn_to_nop(exe: &mut VmExe<F>) {
        for (insn, _) in exe
            .program
            .instructions_and_debug_infos
            .iter_mut()
            .flatten()
        {
            if MulOpcode::iter().any(|op| op.global_opcode() == insn.opcode)
                || MulHOpcode::iter().any(|op| op.global_opcode() == insn.opcode)
                || DivRemOpcode::iter().any(|op| op.global_opcode() == insn.opcode)
            {
                *insn = Instruction::default();
                insn.opcode = SystemOpcode::PHANTOM.global_opcode();
            }
        }
    }
}
