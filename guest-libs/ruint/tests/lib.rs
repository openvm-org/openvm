#[cfg(test)]
mod tests {
    use eyre::Result;
    use openvm_bigint_circuit::{Int256Rv32Config, Int256Rv32Builder};
    use openvm_bigint_transpiler::Int256TranspilerExtension;
    use openvm_circuit::utils::air_test;
    use openvm_instructions::exe::VmExe;
    use openvm_rv32im_transpiler::{
        Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
    };
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use openvm_toolchain_tests::{build_example_program_at_path, get_programs_dir};
    use openvm_transpiler::{transpiler::Transpiler, FromElf};

    type F = BabyBear;

    #[test]
    fn test_matrix_power() -> Result<()> {
        let config = Int256Rv32Config::default();
        let elf = build_example_program_at_path(
            get_programs_dir!("tests/programs"),
            "matrix_power",
            &config,
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(Int256TranspilerExtension),
        )?;
        air_test(Int256Rv32Builder, config, openvm_exe);
        Ok(())
    }
}
