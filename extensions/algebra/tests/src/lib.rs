#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use eyre::Result;
    use num_bigint::BigUint;
    use openvm_algebra_circuit::{
        Fp2Extension, ModularExtension, Rv32ModularConfig, Rv32ModularWithFp2Config,
    };
    use openvm_algebra_transpiler::{Fp2TranspilerExtension, ModularTranspilerExtension};
    use openvm_circuit::{arch::SystemConfig, utils::air_test};
    use openvm_ecc_circuit::SECP256K1_CONFIG;
    use openvm_instructions::exe::VmExe;
    use openvm_rv32im_transpiler::{
        Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
    };
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use openvm_toolchain_tests::{build_example_program_at_path, get_programs_dir, NoInitFile};
    use openvm_transpiler::{transpiler::Transpiler, FromElf};

    type F = BabyBear;

    #[test]
    fn test_moduli_setup() -> Result<()> {
        let moduli = ["4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787", "1000000000000000003", "2305843009213693951"]
            .map(|s| BigUint::from_str(s).unwrap());
        let config = Rv32ModularConfig::new(moduli.to_vec());
        let elf = build_example_program_at_path(get_programs_dir!(), "moduli_setup", &config)?;
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
    fn test_modular() -> Result<()> {
        let config = Rv32ModularConfig::new(vec![SECP256K1_CONFIG.modulus.clone()]);
        let elf = build_example_program_at_path(get_programs_dir!(), "little", &config)?;
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
    fn test_complex_two_moduli() -> Result<()> {
        let config = Rv32ModularWithFp2Config::new(vec![
            (
                "Complex1".to_string(),
                BigUint::from_str("998244353").unwrap(),
            ),
            (
                "Complex2".to_string(),
                BigUint::from_str("1000000007").unwrap(),
            ),
        ]);
        let elf =
            build_example_program_at_path(get_programs_dir!(), "complex_two_moduli", &config)?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(Fp2TranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;
        air_test(config, openvm_exe);
        Ok(())
    }

    #[test]
    fn test_complex_redundant_modulus() -> Result<()> {
        let config = Rv32ModularWithFp2Config {
            system: SystemConfig::default().with_continuations(),
            base: Default::default(),
            mul: Default::default(),
            io: Default::default(),
            modular: ModularExtension::new(vec![
                BigUint::from_str("998244353").unwrap(),
                BigUint::from_str("1000000007").unwrap(),
                BigUint::from_str("1000000009").unwrap(),
                BigUint::from_str("987898789").unwrap(),
            ]),
            fp2: Fp2Extension::new(vec![(
                "Complex2".to_string(),
                BigUint::from_str("1000000009").unwrap(),
            )]),
        };
        let elf = build_example_program_at_path(
            get_programs_dir!(),
            "complex_redundant_modulus",
            &config,
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(Fp2TranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;
        air_test(config, openvm_exe);
        Ok(())
    }

    #[test]
    fn test_complex() -> Result<()> {
        let config = Rv32ModularWithFp2Config::new(vec![(
            "Complex".to_string(),
            SECP256K1_CONFIG.modulus.clone(),
        )]);
        let elf = build_example_program_at_path(get_programs_dir!(), "complex_secp256k1", &config)?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(Fp2TranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;
        air_test(config, openvm_exe);
        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_invalid_setup() {
        let config = Rv32ModularConfig::new(vec![
            BigUint::from_str("998244353").unwrap(),
            BigUint::from_str("1000000007").unwrap(),
        ]);
        let elf = build_example_program_at_path(
            get_programs_dir!(),
            "invalid_setup",
            // We don't want init.rs to be generated for this test because we are testing an
            // invalid moduli_init! call
            &NoInitFile,
        )
        .unwrap();
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(Fp2TranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )
        .unwrap();
        air_test(config, openvm_exe);
    }

    #[test]
    fn test_sqrt() -> Result<()> {
        let config = Rv32ModularConfig::new(vec![SECP256K1_CONFIG.modulus.clone()]);
        let elf = build_example_program_at_path(get_programs_dir!(), "sqrt", &config)?;
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
