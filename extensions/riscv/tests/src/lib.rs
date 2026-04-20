#[cfg(test)]
mod tests {
    #[cfg(feature = "rvr")]
    use std::{env, process::Command};

    use eyre::Result;
    use openvm_circuit::{
        arch::{hasher::poseidon2::vm_poseidon2_hasher, ExecutionError, VmExecutor},
        system::memory::{
            merkle::{public_values::UserPublicValuesProof, MerkleTree},
            online::LinearMemory,
        },
        utils::{air_test, air_test_with_min_segments, test_system_config},
    };
    use openvm_instructions::{
        exe::VmExe, instruction::Instruction, riscv::RV64_REGISTER_NUM_LIMBS, LocalOpcode,
        SystemOpcode,
    };
    use openvm_riscv_circuit::{Rv64IBuilder, Rv64IConfig, Rv64ImBuilder, Rv64ImConfig};
    use openvm_riscv_guest::MAX_HINT_BUFFER_DWORDS;
    use openvm_riscv_transpiler::{
        DivRemOpcode, MulHOpcode, MulOpcode, Rv64ITranspilerExtension, Rv64IoTranspilerExtension,
        Rv64MTranspilerExtension,
    };
    use openvm_stark_sdk::{
        openvm_stark_backend::p3_field::PrimeCharacteristicRing, p3_baby_bear::BabyBear,
    };
    use openvm_toolchain_tests::{
        build_example_program_at_path, build_example_program_at_path_with_features,
        get_programs_dir,
    };
    use openvm_transpiler::{transpiler::Transpiler, FromElf};
    use strum::IntoEnumIterator;
    use test_case::test_case;

    type F = BabyBear;
    #[cfg(feature = "rvr")]
    const RVR_OOB_CHILD_ENV: &str = "OPENVM_RVR_OOB_CHILD";

    #[cfg(test)]
    fn test_rv64im_config() -> Rv64ImConfig {
        Rv64ImConfig {
            rv64i: Rv64IConfig {
                system: test_system_config(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[cfg(feature = "rvr")]
    fn execute_rvr_example(program_name: &str) {
        let config = test_rv64im_config();
        let elf =
            build_example_program_at_path(get_programs_dir!(), program_name, &config).unwrap();
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension),
        )
        .unwrap();
        let executor = VmExecutor::new(config).unwrap();
        let instance = executor.instance(&exe).unwrap();
        instance.execute(vec![], None).unwrap();
    }

    #[cfg(feature = "rvr")]
    fn assert_child_aborts(test_name: &str) {
        let output = Command::new(env::current_exe().unwrap())
            .args(["--exact", test_name, "--nocapture"])
            .env(RVR_OOB_CHILD_ENV, "1")
            .output()
            .expect("failed to spawn self");

        if output.status.success() {
            panic!("child process succeeded; OOB access was not caught");
        }
        // Success path for these tests: relay the child failure text so the
        // caller's `#[should_panic(expected = ...)]` can match it.
        panic!("{}", String::from_utf8_lossy(&output.stderr));
    }

    #[test_case("fibonacci", 1)]
    fn test_rv64i(example_name: &str, min_segments: usize) -> Result<()> {
        let config = Rv64IConfig::default();
        let elf = build_example_program_at_path(get_programs_dir!(), example_name, &config)?;
        let mut exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension),
        )?;
        change_rv64m_insn_to_nop(&mut exe);
        air_test_with_min_segments(Rv64IBuilder, config, exe, vec![], min_segments);
        Ok(())
    }

    #[test]
    fn test_suspend() -> Result<()> {
        let config = test_rv64im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), "fibonacci", &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension),
        )?;

        let executor = VmExecutor::new(config)?;
        let instance = executor.instance(&exe)?;
        let state = instance.execute(vec![], Some(10))?;
        let state = instance.execute_from_state(state, Some(10))?;
        let end_state1 = instance.execute_from_state(state, None)?;
        let end_state2 = instance.execute(vec![], None)?;
        assert_eq!(end_state1.pc(), end_state2.pc());
        for addr_space in 1..end_state1.memory.memory.mem.len() {
            assert_eq!(
                end_state1.memory.memory.mem[addr_space].size(),
                end_state2.memory.memory.mem[addr_space].size()
            );
            let len = end_state2.memory.memory.mem[addr_space].size();
            for i in 0..len {
                unsafe {
                    assert_eq!(
                        end_state1.memory.memory.mem[addr_space].read::<u8>(i),
                        end_state2.memory.memory.mem[addr_space].read::<u8>(i)
                    );
                }
            }
        }
        Ok(())
    }

    #[test_case("fibonacci", 1)]
    #[test_case("collatz", 1)]
    fn test_rv64im(example_name: &str, min_segments: usize) -> Result<()> {
        let config = test_rv64im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), example_name, &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(Rv64MTranspilerExtension),
        )?;
        air_test_with_min_segments(Rv64ImBuilder, config, exe, vec![], min_segments);
        Ok(())
    }

    // TODO(rv64-std): re-enable when guest is updated to no_std
    #[ignore]
    #[test_case("fibonacci", 1)]
    #[test_case("collatz", 1)]
    fn test_rv64im_std(example_name: &str, min_segments: usize) -> Result<()> {
        let config = test_rv64im_config();
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!(),
            example_name,
            ["std"],
            &config,
        )?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(Rv64MTranspilerExtension),
        )?;
        air_test_with_min_segments(Rv64ImBuilder, config, exe, vec![], min_segments);
        Ok(())
    }

    #[test]
    fn test_read_vec() -> Result<()> {
        let config = test_rv64im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), "hint", &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension),
        )?;
        let input = vec![[0, 1, 2, 3].map(F::from_u8).to_vec()];
        air_test_with_min_segments(Rv64ImBuilder, config, exe, input, 1);
        Ok(())
    }

    /// NOTE: This test is slow because it processes > 1MB of data. It is marked #[ignore]
    /// and can be run with: cargo test -p openvm-riscv-integration-tests test_hint_buffer_chunking
    /// -- --ignored
    #[test]
    #[ignore = "slow test: processes >1MB of data"]
    fn test_hint_buffer_chunking() -> Result<()> {
        let config = test_rv64im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), "hint_large_buffer", &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension),
        )?;

        // Create input buffer larger than MAX_HINT_BUFFER_WORDS
        // This will require chunking to succeed
        let expected_words = MAX_HINT_BUFFER_DWORDS + 100;
        let expected_len = expected_words * RV64_REGISTER_NUM_LIMBS;

        // Create data with a pattern that can be verified
        let data: Vec<F> = (0..expected_len)
            .map(|i| F::from_u8((i % 256) as u8))
            .collect();

        let input = vec![data];
        air_test_with_min_segments(Rv64ImBuilder, config, exe, input, 1);
        Ok(())
    }

    #[test]
    fn test_read() -> Result<()> {
        let config = test_rv64im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), "read", &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension),
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
            .map(F::from_u8)
            .collect();
        air_test_with_min_segments(Rv64ImBuilder, config, exe, vec![input], 1);
        Ok(())
    }

    // AOT reaches the same check via fallback, but the panic aborts across its C ABI callback.
    #[cfg_attr(feature = "aot", ignore)]
    #[test]
    #[should_panic(expected = "Memory access out of bounds")]
    fn test_reveal_beyond_num_public_values_errors() {
        let config = test_rv32im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), "reveal", &config).unwrap();
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

    #[test]
    fn test_reveal() -> Result<()> {
        let mut config = test_rv64im_config();
        config.rv64i.system = config.rv64i.system.with_public_values(64);
        let elf = build_example_program_at_path(get_programs_dir!(), "reveal", &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension),
        )?;

        let executor = VmExecutor::new(config.clone())?;
        let instance = executor.instance(&exe)?;
        let state = instance.execute(vec![], None)?;
        let final_memory = state.memory.memory;
        let hasher = vm_poseidon2_hasher::<F>();
        let md = config.as_ref().memory_config.memory_dimensions();
        let tree = MerkleTree::from_memory(&final_memory, &md, &hasher);
        let top_tree = tree.top_tree(md.addr_space_height);
        let pv_proof = UserPublicValuesProof::compute(md, 64, &hasher, &final_memory, &top_tree);
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
                .map(F::from_u8)
                .collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn test_print() -> Result<()> {
        let config = test_rv64im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), "print", &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension),
        )?;
        air_test(Rv64ImBuilder, config, exe);
        Ok(())
    }

    #[test]
    fn test_heap_overflow() -> Result<()> {
        let config = test_rv64im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), "heap_overflow", &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension),
        )?;

        let executor = VmExecutor::new(config)?;
        let instance = executor.instance(&exe)?;
        let input = vec![[0, 0, 0, 1].map(F::from_u8).to_vec()];
        match instance.execute(input.clone(), None) {
            Err(ExecutionError::FailedWithExitCode(_)) => Ok(()),
            Err(_) => panic!("should fail with `FailedWithExitCode`"),
            Ok(_) => panic!("should fail"),
        }
    }

    // TODO(rv64-std): re-enable when guest is updated to no_std
    #[ignore]
    #[test]
    fn test_hashmap() -> Result<()> {
        let config = test_rv64im_config();
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!(),
            "hashmap",
            ["std"],
            &config,
        )?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension),
        )?;
        air_test(Rv64ImBuilder, config, exe);
        Ok(())
    }

    #[test]
    fn test_tiny_mem_test() -> Result<()> {
        let config = test_rv64im_config();
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!(),
            "tiny-mem-test",
            ["heap-embedded-alloc"],
            &config,
        )?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension),
        )?;
        air_test(Rv64ImBuilder, config, exe);
        Ok(())
    }

    #[test]
    #[should_panic]
    // AOT and RVR skip this test since it is not a trusted program: both
    // compile to native code without OpenVM's runtime memory bounds checks,
    // so an out-of-bounds load doesn't surface as a Rust panic.
    #[cfg(all(not(feature = "aot"), not(feature = "rvr")))]
    fn test_load_x0() {
        let config = test_rv64im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), "load_x0", &config).unwrap();
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension),
        )
        .unwrap();
        let executor = VmExecutor::new(config).unwrap();
        let instance = executor.instance(&exe).unwrap();
        instance.execute(vec![], None).unwrap();
    }

    #[test]
    #[should_panic(expected = "Memory access out of bounds")]
    #[cfg(feature = "rvr")]
    fn test_out_of_bound_mem_access() {
        // Child mode: triggers the OOB; abort_oob in C aborts the process.
        if env::var(RVR_OOB_CHILD_ENV).is_ok() {
            execute_rvr_example("out_of_bound_mem_access");
            return; // unreachable: abort fired
        }

        // Parent mode: spawn ourselves as the child and forward its stderr
        // as our own panic. `#[should_panic(expected = ...)]` matches iff
        // the child's rvr bounds check actually fired.
        assert_child_aborts("tests::test_out_of_bound_mem_access");
    }

    #[test]
    #[should_panic(expected = "reveal out of bounds")]
    #[cfg(feature = "rvr")]
    fn test_out_of_bound_reveal() {
        // Child mode: triggers the existing host_reveal public-values bounds assert.
        if env::var(RVR_OOB_CHILD_ENV).is_ok() {
            execute_rvr_example("out_of_bound_reveal");
            return; // unreachable: abort fired
        }

        // Parent mode: spawn ourselves as the child and forward its stderr
        // as our own panic. `#[should_panic(expected = ...)]` matches iff
        // the child's host_reveal assert actually fired.
        assert_child_aborts("tests::test_out_of_bound_reveal");
    }

    #[test]
    #[should_panic(expected = "Memory access out of bounds")]
    #[cfg(feature = "rvr")]
    fn test_out_of_bound_print_str() {
        // Child mode: triggers the Rust-side bounds check in host_print_str.
        if env::var(RVR_OOB_CHILD_ENV).is_ok() {
            execute_rvr_example("out_of_bound_print_str");
            return; // unreachable: abort fired
        }

        // Parent mode: spawn ourselves as the child and forward its stderr
        // as our own panic. `#[should_panic(expected = ...)]` matches iff
        // the child's host_print_str bounds check actually fired.
        assert_child_aborts("tests::test_out_of_bound_print_str");
    }

    #[test_case("getrandom", vec!["getrandom", "getrandom-unsupported"])]
    #[test_case("getrandom", vec!["getrandom"])]
    #[test_case("getrandom_v02", vec!["getrandom-v02", "getrandom-unsupported"])]
    #[test_case("getrandom_v02", vec!["getrandom-v02/custom"])]
    fn test_getrandom_unsupported(program: &str, features: Vec<&str>) {
        let config = test_rv64im_config();
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
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension),
        )
        .unwrap();
        air_test(Rv64ImBuilder, config, exe);
    }

    // For testing programs that should only execute RV64I:
    // The ELF might still have Mul instructions even though the program doesn't use them. We
    // mask those to NOP here.
    fn change_rv64m_insn_to_nop(exe: &mut VmExe<F>) {
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
