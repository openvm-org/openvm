#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use eyre::Result;
    use num_bigint::BigUint;
    use openvm_algebra_circuit::{Rv32ModularConfig, Rv32ModularWithFp2Config};
    use openvm_algebra_transpiler::{Fp2TranspilerExtension, ModularTranspilerExtension};
    use openvm_circuit::utils::air_test;
    use openvm_ecc_circuit::SECP256K1_CONFIG;
    use openvm_instructions::exe::VmExe;
    use openvm_rv32im_transpiler::{
        Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
    };
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use openvm_toolchain_tests::{build_example_program_at_path, get_programs_dir};
    use openvm_transpiler::{transpiler::Transpiler, FromElf};

    type F = BabyBear;

    #[test]
    fn test_moduli_setup() -> Result<()> {
        let elf = build_example_program_at_path(get_programs_dir!(), "moduli_setup")?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;

        let moduli = ["4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787", "1000000000000000003", "2305843009213693951"]
            .map(|s| BigUint::from_str(s).unwrap());
        let config = Rv32ModularConfig::new(moduli.to_vec());
        air_test(config, openvm_exe);
        Ok(())
    }

    #[test]
    fn test_modular() -> Result<()> {
        let elf = build_example_program_at_path(get_programs_dir!(), "little")?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;
        let config = Rv32ModularConfig::new(vec![SECP256K1_CONFIG.modulus.clone()]);
        air_test(config, openvm_exe);
        Ok(())
    }

    #[test]
    fn test_complex_two_moduli() -> Result<()> {
        let elf = build_example_program_at_path(get_programs_dir!(), "complex-two-modulos")?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(Fp2TranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;
        let config = Rv32ModularWithFp2Config::new(vec![
            BigUint::from_str("998244353").unwrap(),
            BigUint::from_str("1000000007").unwrap(),
        ]);
        air_test(config, openvm_exe);
        Ok(())
    }

    #[test]
    fn test_complex() -> Result<()> {
        let elf = build_example_program_at_path(get_programs_dir!(), "complex-secp256k1")?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(Fp2TranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;
        let config = Rv32ModularWithFp2Config::new(vec![SECP256K1_CONFIG.modulus.clone()]);
        air_test(config, openvm_exe);
        Ok(())
    }
}
