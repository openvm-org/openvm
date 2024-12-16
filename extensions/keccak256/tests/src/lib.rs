#[cfg(test)]
mod tests {
    use eyre::Result;
    use openvm_circuit::arch::VmExecutor;
    use openvm_instructions::exe::VmExe;
    use openvm_keccak256_circuit::Keccak256Rv32Config;
    use openvm_keccak256_transpiler::Keccak256TranspilerExtension;
    use openvm_rv32im_transpiler::{
        Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
    };
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use openvm_toolchain_tests::{build_example_program_at_path, get_programs_dir};
    use openvm_transpiler::{transpiler::Transpiler, FromElf};

    type F = BabyBear;

    #[test]
    fn test_keccak256_runtime() -> Result<()> {
        let elf = build_example_program_at_path(get_programs_dir!(), "keccak")?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Keccak256TranspilerExtension)
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;
        let executor = VmExecutor::<F, Keccak256Rv32Config>::new(Keccak256Rv32Config::default());
        executor.execute(openvm_exe, vec![])?;
        Ok(())
    }
}
