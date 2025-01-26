#[cfg(test)]
mod tests {
    use eyre::Result;
    use openvm_build::{GuestOptions, TargetFilter};
    use openvm_circuit::{arch::instructions::exe::VmExe, utils::air_test};
    use openvm_native_circuit::Rv32WithKernelsConfig;
    use openvm_native_transpiler::LongFormTranspilerExtension;
    use openvm_rv32im_transpiler::{
        Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
    };
    use openvm_sdk::Sdk;
    use openvm_toolchain_tests::get_programs_dir;
    use openvm_transpiler::{transpiler::Transpiler, FromElf};
    use p3_baby_bear::BabyBear;

    #[test]
    fn test_native_kernel() -> Result<()> {
        let sdk = Sdk;

        let elf = sdk.build(
            GuestOptions::default(),
            get_programs_dir!(),
            &Some(TargetFilter {
                kind: "bin".to_string(),
                name: "openvm-native-integration-test-program".to_string(),
            }),
        )?;
        let exe = VmExe::from_elf(
            elf.clone(),
            Transpiler::<BabyBear>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(LongFormTranspilerExtension),
        )?;

        air_test(Rv32WithKernelsConfig::default(), exe);
        Ok(())
    }
}
