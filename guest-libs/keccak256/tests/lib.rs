#[cfg(test)]
mod tests {
    use eyre::Result;
    use openvm_circuit::utils::air_test;
    use openvm_instructions::exe::VmExe;
    use openvm_keccak256_circuit::Keccak256Rv32Config;
    #[cfg(not(feature = "cuda"))]
    use openvm_keccak256_circuit::Keccak256Rv32CpuBuilder as TestBuilder;
    #[cfg(feature = "cuda")]
    use openvm_keccak256_circuit::Keccak256Rv32GpuBuilder as TestBuilder;
    use openvm_keccak256_transpiler::Keccak256TranspilerExtension;
    use openvm_riscv_transpiler::{
        Rv64ITranspilerExtension, Rv64IoTranspilerExtension, Rv64MTranspilerExtension,
    };
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use openvm_toolchain_tests::{build_example_program_at_path, get_programs_dir};
    use openvm_transpiler::{transpiler::Transpiler, FromElf};

    type F = BabyBear;

    #[test]
    fn test_keccak256() -> Result<()> {
        let config = Keccak256Rv32Config::default();
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
        air_test(TestBuilder, config, openvm_exe);
        Ok(())
    }
}
