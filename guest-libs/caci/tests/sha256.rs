#[cfg(test)]
mod tests {
    use eyre::Result;
    use openvm_circuit::{arch::VmExecutor, utils::air_test_with_min_segments};
    use openvm_instructions::exe::VmExe;
    use openvm_riscv_transpiler::{
        Rv64ITranspilerExtension, Rv64IoTranspilerExtension, Rv64MTranspilerExtension,
    };
    use openvm_sdk::StdIn;
    use openvm_sha2_circuit::{Sha2Rv64Builder, Sha2Rv64Config};
    use openvm_sha2_transpiler::Sha2TranspilerExtension;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use openvm_toolchain_tests::{build_example_program_at_path, get_programs_dir};
    use openvm_transpiler::{transpiler::Transpiler, FromElf};
    use sha2::{Digest, Sha256};

    type F = BabyBear;

    fn reference_sha256(input: &[u8]) -> [u8; 32] {
        Sha256::digest(input).into()
    }

    /// Input lengths chosen around the SHA-256 block size (64 bytes) and the padding boundary
    /// (55/56 bytes): empty, sub-block, boundary +/- 1, and multi-block.
    const INPUT_LENS: &[usize] = &[0, 1, 32, 55, 56, 63, 64, 65, 128, 1000];

    fn test_zkvm_sha256_base(prove: bool) -> Result<()> {
        let config = Sha2Rv64Config::default();
        let elf =
            build_example_program_at_path(get_programs_dir!("tests/programs"), "sha256", &config)?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(Sha2TranspilerExtension),
        )?;

        let mut stdin = StdIn::default();
        stdin.write(&(INPUT_LENS.len() as u32));
        for (i, &len) in INPUT_LENS.iter().enumerate() {
            let input: Vec<u8> = (0..len).map(|j| (i + j) as u8).collect();
            stdin.write(&input);
            stdin.write(&reference_sha256(&input).to_vec());
        }

        if prove {
            air_test_with_min_segments(Sha2Rv64Builder, config, openvm_exe, stdin, 1);
        } else {
            let executor = VmExecutor::new(config)?;
            let instance = executor.instance(&openvm_exe)?;
            instance.execute(stdin)?;
        }
        Ok(())
    }

    #[test]
    fn test_zkvm_sha256_run() -> Result<()> {
        test_zkvm_sha256_base(false)
    }

    #[test]
    #[ignore = "proving on CPU is slow"]
    fn test_zkvm_sha256_prove() -> Result<()> {
        test_zkvm_sha256_base(true)
    }
}
