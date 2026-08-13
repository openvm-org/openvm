#[cfg(test)]
mod tests {
    use std::fs;

    use eyre::Result;
    #[cfg(feature = "aot")]
    use openvm_circuit::arch::{testing::assert_vm_states_equivalent, SystemConfig};
    use openvm_circuit::{arch::VmExecutor, utils::air_test_with_min_segments};
    use openvm_instructions::exe::VmExe;
    use openvm_poseidon2::{hash_bytes, hash_u32s};
    use openvm_poseidon2_circuit::{Poseidon2Rv32Config, Poseidon2Rv32CpuBuilder};
    use openvm_poseidon2_transpiler::Poseidon2TranspilerExtension;
    use openvm_rv32im_transpiler::{
        Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
    };
    use openvm_sdk::StdIn;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use openvm_toolchain_tests::{build_example_program_at_path, get_programs_dir};
    use openvm_transpiler::{transpiler::Transpiler, FromElf};
    use rand::RngCore;

    type F = BabyBear;

    struct TestVector {
        input: Vec<u32>,
        expected_output: Vec<u32>,
    }

    const KAT_FILE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/test_vectors/poseidon2_kat.txt"
    );

    fn parse_test_vectors() -> Vec<TestVector> {
        let file_content = fs::read_to_string(KAT_FILE).unwrap();
        let lines = file_content
            .lines()
            .filter(|line| !line.trim_start().starts_with('#'));
        let mut nums = lines
            .flat_map(|line| line.split_whitespace())
            .map(|token| token.parse::<u32>().unwrap());

        let num_test_vectors = nums.next().unwrap() as usize;
        let mut test_vectors = Vec::with_capacity(num_test_vectors);
        for _ in 0..num_test_vectors {
            let len = nums.next().unwrap() as usize;
            let input = nums.by_ref().take(len).collect();
            let expected_output = nums.by_ref().take(openvm_poseidon2::DIGEST_SIZE).collect();
            test_vectors.push(TestVector {
                input,
                expected_output,
            });
        }
        test_vectors
    }

    fn test_poseidon2_base(prove: bool) -> Result<()> {
        let config = Poseidon2Rv32Config::default();
        let elf = build_example_program_at_path(
            get_programs_dir!("tests/programs"),
            "poseidon2",
            &config,
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Poseidon2TranspilerExtension)
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;

        let test_vectors = parse_test_vectors();
        let mut stdin = StdIn::default();
        stdin.write(&(test_vectors.len() as u32));
        for test_vector in &test_vectors {
            stdin.write(&test_vector.input);
            stdin.write(&test_vector.expected_output);
        }

        if prove {
            air_test_with_min_segments(Poseidon2Rv32CpuBuilder, config, openvm_exe, stdin, 1);
        } else {
            let executor = VmExecutor::new(config.clone())?;
            let interpreter = executor.instance(&openvm_exe)?;
            #[allow(unused_variables)]
            let state = interpreter.execute(stdin.clone(), None)?;

            #[cfg(feature = "aot")]
            {
                let naive_interpreter = executor.interpreter_instance(&openvm_exe)?;
                let naive_state = naive_interpreter.execute(stdin, None)?;
                let system_config: &SystemConfig = config.as_ref();
                assert_vm_states_equivalent(
                    &state,
                    &naive_state,
                    &system_config.memory_config.memory_dimensions(),
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_poseidon2_run() -> Result<()> {
        test_poseidon2_base(false)
    }

    #[test]
    #[ignore = "proving on CPU is slow"]
    fn test_poseidon2_prove() -> Result<()> {
        test_poseidon2_base(true)
    }

    /// `hash_bytes` must be total. It previously packed 4 bytes per field element and asserted
    /// canonicality, so any word >= `0x78000001` aborted execution — 53% of uniformly random words,
    /// i.e. essentially every real 32-byte input.
    #[test]
    fn hash_bytes_accepts_arbitrary_input() {
        for len in [0usize, 1, 2, 3, 4, 5, 8, 31, 32, 33, 64] {
            for fill in [0x00u8, 0x01, 0x7f, 0xff] {
                let input = vec![fill; len];
                // Must not panic, and must not depend on word canonicality.
                let _ = hash_bytes(&input);
            }
        }
    }

    /// The byte-level `pad10*1` in `hash_bytes` exists so that inputs differing only in trailing
    /// zeroes do not pack to identical field elements.
    #[test]
    fn hash_bytes_distinguishes_trailing_zeroes() {
        let cases: [(&[u8], &[u8]); 4] = [
            (&[], &[0x00]),
            (&[0x07], &[0x07, 0x00]),
            // Boundary: the left side exactly fills a 3-byte group, so the padding group is added
            // on its own.
            (&[0x01, 0x02, 0x03], &[0x01, 0x02, 0x03, 0x00]),
            (&[0x01, 0x02, 0x03], &[0x01, 0x02, 0x03, 0x00, 0x00, 0x00]),
        ];
        for (left, right) in cases {
            assert_ne!(
                hash_bytes(left),
                hash_bytes(right),
                "hash_bytes collided on {left:?} and {right:?}"
            );
        }
    }

    /// Regenerates the committed KAT file from the host (plonky3 `default_babybear_poseidon2_16`)
    /// reference implementation, which matches the permutation constrained by the `PERMUTE`
    /// circuit.
    /// Run with `cargo nextest run -p openvm-poseidon2 --run-ignored=only
    /// regenerate_poseidon2_kat`.
    #[test]
    #[ignore = "regenerates the committed KAT file"]
    fn regenerate_poseidon2_kat() {
        use openvm_stark_sdk::utils::create_seeded_rng_with_seed;

        let mut rng = create_seeded_rng_with_seed(0);
        let lengths = [0usize, 1, 7, 8, 9, 16, 17, 32];

        let mut content = String::new();
        content.push_str("# poseidon2 hash_u32s KATs, generated via the plonky3 host reference.\n");
        content.push_str("# format: <len> <input words...> <expected digest words...>\n");
        content.push_str(&format!("{}\n", lengths.len()));
        for &len in &lengths {
            let input: Vec<u32> = (0..len).map(|_| rng.next_u32() % (1 << 30)).collect();
            let digest = hash_u32s(&input);
            content.push_str(&len.to_string());
            for word in &input {
                content.push_str(&format!(" {word}"));
            }
            for word in &digest {
                content.push_str(&format!(" {word}"));
            }
            content.push('\n');
        }
        let kat_path = std::path::Path::new(KAT_FILE);
        if let Some(dir) = kat_path.parent() {
            fs::create_dir_all(dir).unwrap();
        }
        fs::write(kat_path, content).unwrap();
    }
}
