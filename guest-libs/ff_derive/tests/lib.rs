#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use eyre::Result;
    use num_bigint::BigUint;
    use openvm_algebra_circuit::Rv32ModularConfig;
    use openvm_algebra_transpiler::ModularTranspilerExtension;
    use openvm_circuit::utils::air_test;
    use openvm_instructions::exe::VmExe;
    use openvm_rv32im_transpiler::{
        Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
    };
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use openvm_toolchain_tests::{
        build_example_program_at_path, build_example_program_at_path_with_features,
        get_programs_dir,
    };
    use openvm_transpiler::{transpiler::Transpiler, FromElf};

    type F = BabyBear;

    #[test]
    fn test_full_limbs() -> Result<()> {
        let moduli = ["39402006196394479212279040100143613805079739270465446667948293404245721771496870329047266088258938001861606973112319"]
        .map(|s| BigUint::from_str(s).unwrap());
        let config = Rv32ModularConfig::new(moduli.to_vec());
        let elf = build_example_program_at_path(
            get_programs_dir!("tests/programs"),
            "full_limbs",
            &config,
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;

        air_test(config, openvm_exe);
        Ok(())
    }

    #[test]
    fn test_fermat() -> Result<()> {
        let moduli = ["65537"].map(|s| BigUint::from_str(s).unwrap());
        let config = Rv32ModularConfig::new(moduli.to_vec());
        let elf =
            build_example_program_at_path(get_programs_dir!("tests/programs"), "fermat", &config)?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;

        air_test(config, openvm_exe);
        Ok(())
    }

    #[test]
    fn test_sqrt() -> Result<()> {
        let moduli = ["357686312646216567629137"].map(|s| BigUint::from_str(s).unwrap());
        let config = Rv32ModularConfig::new(moduli.to_vec());
        let elf =
            build_example_program_at_path(get_programs_dir!("tests/programs"), "sqrt", &config)?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;

        air_test(config, openvm_exe);
        Ok(())
    }

    #[test]
    fn test_constants() -> Result<()> {
        let moduli =
            ["52435875175126190479447740508185965837690552500527637822603658699938581184513"]
                .map(|s| BigUint::from_str(s).unwrap());
        let config = Rv32ModularConfig::new(moduli.to_vec());
        let elf = build_example_program_at_path(
            get_programs_dir!("tests/programs"),
            "constants",
            &config,
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;

        air_test(config, openvm_exe);
        Ok(())
    }

    #[test]
    fn test_from_u128() -> Result<()> {
        let moduli =
            ["52435875175126190479447740508185965837690552500527637822603658699938581184513"]
                .map(|s| BigUint::from_str(s).unwrap());
        let config = Rv32ModularConfig::new(moduli.to_vec());
        let elf = build_example_program_at_path(
            get_programs_dir!("tests/programs"),
            "from_u128",
            &config,
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;

        air_test(config, openvm_exe);
        Ok(())
    }

    #[test]
    fn test_batch_inversion() -> Result<()> {
        let moduli =
            ["52435875175126190479447740508185965837690552500527637822603658699938581184513"]
                .map(|s| BigUint::from_str(s).unwrap());
        let config = Rv32ModularConfig::new(moduli.to_vec());
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!("tests/programs"),
            "batch_inversion",
            ["std"],
            &config,
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;

        air_test(config, openvm_exe);
        Ok(())
    }

    #[test]
    fn test_operations() -> Result<()> {
        let moduli =
            ["52435875175126190479447740508185965837690552500527637822603658699938581184513"]
                .map(|s| BigUint::from_str(s).unwrap());
        let config = Rv32ModularConfig::new(moduli.to_vec());
        let elf = build_example_program_at_path(
            get_programs_dir!("tests/programs"),
            "operations",
            &config,
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;

        air_test(config, openvm_exe);
        Ok(())
    }
}
