#[cfg(test)]
mod tests {
    use eyre::Result;
    use openvm_bigint_circuit::Int256Rv32Config;
    use openvm_bigint_transpiler::Int256TranspilerExtension;
    use openvm_circuit::arch::VmExecutor;
    use openvm_instructions::exe::VmExe;
    use openvm_rv32im_transpiler::{
        Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
    };
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use openvm_toolchain_tests::{build_example_program_at_path, get_programs_dir};
    use openvm_transpiler::{transpiler::Transpiler, FromElf};

    type F = BabyBear;

    #[test]
    fn test_matrix_power_runtime() -> Result<()> {
        let elf = build_example_program_at_path(get_programs_dir!(), "matrix-power")?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(Int256TranspilerExtension),
        )?;
        let config = Int256Rv32Config::default();
        let executor = VmExecutor::<F, _>::new(config);
        executor.execute(openvm_exe, vec![])?;
        Ok(())
    }

    #[test]
    fn test_matrix_power_signed_runtime() -> Result<()> {
        let elf = build_example_program_at_path(get_programs_dir!(), "matrix-power-signed")?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(Int256TranspilerExtension),
        )?;
        let config = Int256Rv32Config::default();
        let executor = VmExecutor::<F, _>::new(config);
        executor.execute(openvm_exe, vec![])?;
        Ok(())
    }
}
