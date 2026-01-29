#[cfg(test)]
mod tests {
    use std::fs;

    use eyre::Result;
    use hex::FromHex;
    use openvm_circuit::{arch::VmExecutor, utils::air_test_with_min_segments};
    use openvm_instructions::exe::VmExe;
    use openvm_rv32im_transpiler::{
        Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
    };
    use openvm_sdk::StdIn;
    use openvm_sha2_circuit::{Sha2Rv32Builder, Sha2Rv32Config};
    use openvm_sha2_transpiler::Sha2TranspilerExtension;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use openvm_toolchain_tests::{build_example_program_at_path, get_programs_dir};
    use openvm_transpiler::{transpiler::Transpiler, FromElf};

    type F = BabyBear;

    enum Sha2Type {
        Sha256,
        Sha384,
        Sha512,
    }

    struct TestVector {
        input: Vec<u8>,
        expected_output: Vec<u8>,
    }

    // test vectors are from https://csrc.nist.gov/projects/cryptographic-algorithm-validation-program/secure-hashing
    fn parse_test_vectors(file_name: &str) -> Vec<TestVector> {
        let mut test_vectors = Vec::new();

        let get_attribute_from_line = |line: &str, attribute: &str| -> Option<String> {
            line.trim()
                .strip_prefix(&(attribute.to_string() + " = "))
                .map(|s| s.to_string())
        };

        let file_content =
            fs::read_to_string("tests/programs/examples/test_vectors/".to_string() + file_name)
                .unwrap();
        let mut lines = file_content.lines();
        while let Some(line) = lines.next() {
            let Some(len_str) = get_attribute_from_line(line, "Len") else {
                continue;
            };
            let msg_str = get_attribute_from_line(lines.next().unwrap(), "Msg").unwrap();
            let md_str = get_attribute_from_line(lines.next().unwrap(), "MD").unwrap();

            let len: usize = len_str.parse().unwrap();
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

        test_vectors
    }

    fn test_sha2_base(test_vector_file_name: &str, sha2_type: Sha2Type, prove: bool) -> Result<()> {
        let config = Sha2Rv32Config::default();
        let elf =
            build_example_program_at_path(get_programs_dir!("tests/programs"), "sha2", &config)?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(Sha2TranspilerExtension),
        )?;

        for test_vector in parse_test_vectors(test_vector_file_name) {
            let mut stdin = StdIn::default();
            stdin.write(&match sha2_type {
                Sha2Type::Sha256 => 256u32,
                Sha2Type::Sha384 => 384u32,
                Sha2Type::Sha512 => 512u32,
            });
            stdin.write(&test_vector.input);
            stdin.write(&test_vector.expected_output);

            if prove {
                air_test_with_min_segments(
                    Sha2Rv32Builder,
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

    #[test]
    fn test_sha256_run() -> Result<()> {
        test_sha2_base("SHA256ShortMsg.rsp", Sha2Type::Sha256, false)?;
        test_sha2_base("SHA256LongMsg.rsp", Sha2Type::Sha256, false)
    }

    #[test]
    fn test_sha384_run() -> Result<()> {
        test_sha2_base("SHA384ShortMsg.rsp", Sha2Type::Sha384, false)?;
        test_sha2_base("SHA384LongMsg.rsp", Sha2Type::Sha384, false)
    }

    #[test]
    fn test_sha512_run() -> Result<()> {
        test_sha2_base("SHA512ShortMsg.rsp", Sha2Type::Sha512, false)?;
        test_sha2_base("SHA512LongMsg.rsp", Sha2Type::Sha512, false)
    }

    #[test]
    #[ignore = "proving on CPU is slow"]
    fn test_sha256_prove() -> Result<()> {
        test_sha2_base("SHA256ShortMsg.rsp", Sha2Type::Sha256, true)?;
        #[cfg(feature = "long_proving_tests")]
        test_sha2_base("SHA256LongMsg.rsp", Sha2Type::Sha256, true)?;
        Ok(())
    }

    #[test]
    #[ignore = "proving on CPU is slow"]
    fn test_sha384_prove() -> Result<()> {
        test_sha2_base("SHA384ShortMsg.rsp", Sha2Type::Sha384, true)?;
        #[cfg(feature = "long_proving_tests")]
        test_sha2_base("SHA384LongMsg.rsp", Sha2Type::Sha384, true)?;
        Ok(())
    }

    #[test]
    #[ignore = "proving on CPU is slow"]
    fn test_sha512_prove() -> Result<()> {
        test_sha2_base("SHA512ShortMsg.rsp", Sha2Type::Sha512, true)?;
        #[cfg(feature = "long_proving_tests")]
        test_sha2_base("SHA512LongMsg.rsp", Sha2Type::Sha512, true)?;
        Ok(())
    }
}
