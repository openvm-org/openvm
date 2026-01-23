#[cfg(test)]
mod tests {
    use eyre::Result;
    use hex::FromHex;
    use openvm_circuit::{arch::Streams, utils::air_test_with_min_segments};
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
    use std::fs;

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

    fn test_sha2(test_vector_file_name: &str, sha2_type: Sha2Type) -> Result<()> {
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

            air_test_with_min_segments(
                Sha2Rv32Builder,
                config.clone(),
                openvm_exe.clone(),
                <Streams<F> as From<StdIn<F>>>::from(stdin),
                1,
            );
        }

        Ok(())
    }

    #[test]
    fn test_sha256_short() -> Result<()> {
        test_sha2("SHA256ShortMsg.rsp", Sha2Type::Sha256)
    }
}
