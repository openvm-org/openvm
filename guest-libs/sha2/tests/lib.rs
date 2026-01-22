#[cfg(test)]
mod tests {
    use eyre::Result;
    use hex::FromHex;
    use openvm_circuit::arch::Streams;
    use openvm_circuit::utils::air_test_with_min_segments;
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

    #[test]
    fn test_sha2() -> Result<()> {
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
        let mut stdin = StdIn::default();
        let input_string = "";
        let output_string = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

        stdin.write(&Vec::from_hex(input_string).unwrap());
        stdin.write(&Vec::from_hex(output_string).unwrap());
        air_test_with_min_segments(
            Sha2Rv32Builder,
            config,
            openvm_exe,
            <Streams<F> as From<StdIn<F>>>::from(stdin),
            1,
        );
        Ok(())
    }
}
