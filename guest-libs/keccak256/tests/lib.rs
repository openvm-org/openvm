#[cfg(test)]
mod tests {
    use std::fs;

    use eyre::Result;
    use hex::FromHex;
    use openvm_circuit::{arch::VmExecutor, utils::air_test_with_min_segments};
    use openvm_instructions::exe::VmExe;
    use openvm_keccak256_circuit::Keccak256Rv32Config;
    #[cfg(not(feature = "cuda"))]
    use openvm_keccak256_circuit::Keccak256Rv32CpuBuilder as TestBuilder;
    #[cfg(feature = "cuda")]
    use openvm_keccak256_circuit::Keccak256Rv32GpuBuilder as TestBuilder;
    use openvm_keccak256_transpiler::Keccak256TranspilerExtension;
    use openvm_rv32im_transpiler::{
        Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
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
    fn parse_test_vectors(file_name: &str, len_filter: impl Fn(usize) -> bool) -> Vec<TestVector> {
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
            if len.is_multiple_of(8) && len_filter(len / 8) {
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

    fn test_keccak256_base(
        test_vector_file_name: &str,
        prove: bool,
        len_mod4_filter: usize,
    ) -> Result<()> {
        let config = Keccak256Rv32Config::default();
        let elf =
            build_example_program_at_path(get_programs_dir!("tests/programs"), "keccak", &config)?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Keccak256TranspilerExtension)
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;

        for test_vector in
            parse_test_vectors(test_vector_file_name, |len| len % 4 == len_mod4_filter)
        {
            let mut stdin = StdIn::default();
            stdin.write(&test_vector.input);
            stdin.write(&test_vector.expected_output);

            if prove {
                air_test_with_min_segments(
                    TestBuilder,
                    config.clone(),
                    openvm_exe.clone(),
                    stdin,
                    1,
                );
            } else {
                let executor = VmExecutor::new(config.clone())?;
                let interpreter = executor.instance(&openvm_exe)?;
                #[allow(unused_variables)]
                let state = interpreter.execute(stdin.clone(), None)?;

                #[cfg(feature = "aot")]
                {
                    use openvm_circuit::{arch::VmState, system::memory::online::GuestMemory};
                    let naive_interpreter = executor.interpreter_instance(&openvm_exe)?;
                    let naive_state = naive_interpreter.execute(stdin, None)?;
                    let assert_vm_state_eq =
                        |lhs: &VmState<BabyBear, GuestMemory>,
                         rhs: &VmState<BabyBear, GuestMemory>| {
                            assert_eq!(lhs.pc(), rhs.pc());
                            for r in 0..32 {
                                let a = unsafe { lhs.memory.read::<u8, 1>(1, r as u32) };
                                let b = unsafe { rhs.memory.read::<u8, 1>(1, r as u32) };
                                assert_eq!(a, b);
                            }
                        };
                    assert_vm_state_eq(&state, &naive_state);
                }
            }
        }

        Ok(())
    }

    fn test_keccak256_run_base(len_mod4_filter: usize) -> Result<()> {
        test_keccak256_base("ShortMsgKAT_256.txt", false, len_mod4_filter)?;
        test_keccak256_base("LongMsgKAT_256.txt", false, len_mod4_filter)
    }

    #[test]
    fn test_keccak256_run_0mod4() -> Result<()> {
        test_keccak256_run_base(0)
    }

    #[test]
    fn test_keccak256_run_1mod4() -> Result<()> {
        test_keccak256_run_base(1)
    }

    #[test]
    fn test_keccak256_run_2mod4() -> Result<()> {
        test_keccak256_run_base(2)
    }

    #[test]
    fn test_keccak256_run_3mod4() -> Result<()> {
        test_keccak256_run_base(3)
    }

    fn test_keccak256_prove_base(len_mod4_filter: usize) -> Result<()> {
        test_keccak256_base("ShortMsgKAT_256.txt", true, len_mod4_filter)?;
        #[cfg(feature = "long_proving_tests")]
        test_keccak256_base("LongMsgKAT_256.txt", true, len_mod4_filter)?;
        Ok(())
    }

    #[test]
    #[ignore = "proving on CPU is slow"]
    fn test_keccak256_prove_0mod4() -> Result<()> {
        test_keccak256_prove_base(0)
    }

    #[test]
    #[ignore = "proving on CPU is slow"]
    fn test_keccak256_prove_1mod4() -> Result<()> {
        test_keccak256_prove_base(1)
    }

    #[test]
    #[ignore = "proving on CPU is slow"]
    fn test_keccak256_prove_2mod4() -> Result<()> {
        test_keccak256_prove_base(2)
    }

    #[test]
    #[ignore = "proving on CPU is slow"]
    fn test_keccak256_prove_3mod4() -> Result<()> {
        test_keccak256_prove_base(3)
    }
}
