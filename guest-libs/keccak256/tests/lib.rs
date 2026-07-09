#[cfg(test)]
mod tests {
    use std::fs;

    use eyre::Result;
    use hex::FromHex;
    #[cfg(feature = "rvr")]
    use openvm_circuit::arch::testing::assert_vm_states_equivalent;
    use openvm_circuit::{arch::VmExecutor, utils::air_test_with_min_segments};
    use openvm_instructions::exe::VmExe;
    use openvm_keccak256_circuit::Keccak256Rv64Config;
    #[cfg(not(feature = "cuda"))]
    use openvm_keccak256_circuit::Keccak256Rv64CpuBuilder as TestBuilder;
    #[cfg(feature = "cuda")]
    use openvm_keccak256_circuit::Keccak256Rv64GpuBuilder as TestBuilder;
    use openvm_keccak256_transpiler::Keccak256TranspilerExtension;
    use openvm_riscv_transpiler::{
        Rv64ITranspilerExtension, Rv64IoTranspilerExtension, Rv64MTranspilerExtension,
    };
    use openvm_sdk::StdIn;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use openvm_toolchain_tests::{build_example_program_at_path, get_programs_dir};
    use openvm_transpiler::{transpiler::Transpiler, FromElf};

    type F = BabyBear;

    struct TestVector {
        input: Vec<u8>,
        expected_output: Vec<u8>,
    }

    // test vectors are from https://keccak.team/archives.html (https://keccak.team/obsolete/KeccakKAT-3.zip)
    fn parse_test_vectors(file_name: &str) -> Vec<TestVector> {
        let mut test_vectors = Vec::new();

        let get_attribute_from_line = |line: &str, attribute: &str| -> Option<String> {
            line.trim()
                .strip_prefix(&(attribute.to_string() + " = "))
                .map(|s| s.to_string())
        };

        let file_content =
            fs::read_to_string("tests/test_vectors/".to_string() + file_name).unwrap();
        let mut lines = file_content.lines();
        while let Some(line) = lines.next() {
            let Some(len_str) = get_attribute_from_line(line, "Len") else {
                continue;
            };
            let msg_str = get_attribute_from_line(lines.next().unwrap(), "Msg").unwrap();
            let md_str = get_attribute_from_line(lines.next().unwrap(), "MD").unwrap();

            let len: usize = len_str.parse().unwrap();
            if len.is_multiple_of(8) {
                let msg = if len == 0 {
                    Vec::new()
                } else {
                    Vec::from_hex(&msg_str).unwrap()
                };
                let md = Vec::from_hex(&md_str).unwrap();

                test_vectors.push(TestVector {
                    input: msg,
                    expected_output: md,
                });
            }
        }

        test_vectors
    }

    fn test_keccak256_base(test_vector_file_name: &str, prove: bool) -> Result<()> {
        let config = Keccak256Rv64Config::default();
        let elf =
            build_example_program_at_path(get_programs_dir!("tests/programs"), "keccak", &config)?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Keccak256TranspilerExtension)
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension),
        )?;

        let test_vectors = parse_test_vectors(test_vector_file_name);
        let mut stdin = StdIn::default();
        stdin.write(&(test_vectors.len() as u32));
        for test_vector in &test_vectors {
            stdin.write(&test_vector.input);
            stdin.write(&test_vector.expected_output);
        }

        if prove {
            air_test_with_min_segments(TestBuilder, config, openvm_exe, stdin, 1);
        } else {
            let executor = VmExecutor::new(config.clone())?;
            let instance = executor.instance(&openvm_exe)?;
            #[allow(unused_variables)]
            let state = instance.execute(stdin.clone())?;

            #[cfg(feature = "rvr")]
            {
                let interpreter_instance = executor.interpreter_instance(&openvm_exe)?;
                let naive_state = interpreter_instance.execute(stdin)?;
                assert_vm_states_equivalent(&state, &naive_state);
            }
        }

        Ok(())
    }

    #[test]
    fn test_keccak256_run() -> Result<()> {
        test_keccak256_base("ShortMsgKAT_256.txt", false)?;
        test_keccak256_base("LongMsgKAT_256.txt", false)
    }

    #[cfg(all(feature = "rvr", not(feature = "cuda")))]
    #[test]
    fn test_keccak256_rvr_preflight_prove() -> Result<()> {
        let config = Keccak256Rv64Config::default();
        let elf = build_example_program_at_path(
            get_programs_dir!("tests/programs"),
            "keccak_rvr",
            &config,
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Keccak256TranspilerExtension)
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension),
        )?;
        air_test_with_min_segments(TestBuilder, config, openvm_exe, StdIn::default(), 1);
        Ok(())
    }

    #[test]
    #[ignore = "proving on CPU is slow"]
    fn test_keccak256_prove() -> Result<()> {
        test_keccak256_base("ShortMsgKAT_256.txt", true)?;
        test_keccak256_base("LongMsgKAT_256.txt", true)
    }
}
