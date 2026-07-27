#[cfg(test)]
mod tests {
    use eyre::Result;
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
    use tiny_keccak::{Hasher, Keccak};

    type F = BabyBear;

    fn reference_keccak256(input: &[u8]) -> [u8; 32] {
        let mut hasher = Keccak::v256();
        hasher.update(input);
        let mut output = [0u8; 32];
        hasher.finalize(&mut output);
        output
    }

    /// Input lengths chosen around the sponge rate (136 bytes): empty, sub-block, block
    /// boundary +/- 1, and multi-block.
    const INPUT_LENS: &[usize] = &[0, 1, 32, 135, 136, 137, 272, 1000];

    fn test_zkvm_keccak256_base(prove: bool) -> Result<()> {
        let config = Keccak256Rv64Config::default();
        let elf = build_example_program_at_path(
            get_programs_dir!("tests/programs"),
            "keccak256",
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

        let mut stdin = StdIn::default();
        stdin.write(&(INPUT_LENS.len() as u32));
        for (i, &len) in INPUT_LENS.iter().enumerate() {
            let input: Vec<u8> = (0..len).map(|j| (i + j) as u8).collect();
            stdin.write(&input);
            stdin.write(&reference_keccak256(&input).to_vec());
        }

        if prove {
            air_test_with_min_segments(TestBuilder, config, openvm_exe, stdin, 1);
        } else {
            let executor = VmExecutor::new(config)?;
            let instance = executor.instance(&openvm_exe)?;
            instance.execute(stdin)?;
        }
        Ok(())
    }

    #[test]
    fn test_zkvm_keccak256_run() -> Result<()> {
        test_zkvm_keccak256_base(false)
    }

    #[test]
    #[ignore = "proving on CPU is slow"]
    fn test_zkvm_keccak256_prove() -> Result<()> {
        test_zkvm_keccak256_base(true)
    }
}
