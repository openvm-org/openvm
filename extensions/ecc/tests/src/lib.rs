#[cfg(test)]
mod tests {
    use eyre::Result;
    use openvm_algebra_circuit::ModularExtension;
    use openvm_algebra_transpiler::ModularTranspilerExtension;
    use openvm_circuit::{
        arch::{instructions::exe::VmExe, SystemConfig},
        utils::{air_test, air_test_with_min_segments},
    };
    use openvm_ecc_circuit::{
        Rv32WeierstrassConfig, WeierstrassExtension, P256_CONFIG, SECP256K1_CONFIG,
    };
    use openvm_ecc_transpiler::EccTranspilerExtension;
    use openvm_keccak256_transpiler::Keccak256TranspilerExtension;
    use openvm_rv32im_transpiler::{
        Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
    };
    use openvm_sdk::config::SdkVmConfig;
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::{openvm_stark_backend, p3_baby_bear::BabyBear};
    use openvm_toolchain_tests::{build_example_program_at_path_with_features, get_programs_dir};
    use openvm_transpiler::{transpiler::Transpiler, FromElf};
    type F = BabyBear;

    #[test]
    fn test_ec() -> Result<()> {
        let elf = build_example_program_at_path_with_features(get_programs_dir!(), "ec", ["k256"])?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(EccTranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;
        let config = Rv32WeierstrassConfig::new(vec![SECP256K1_CONFIG.clone()]);
        air_test(config, openvm_exe);
        Ok(())
    }

    #[test]
    fn test_ec_nonzero_a() -> Result<()> {
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!(),
            "ec_nonzero_a",
            ["p256"],
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(EccTranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;
        let config = Rv32WeierstrassConfig::new(vec![P256_CONFIG.clone()]);
        air_test(config, openvm_exe);
        Ok(())
    }

    #[test]
    fn test_ec_two_curves() -> Result<()> {
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!(),
            "ec_two_curves",
            ["k256", "p256"],
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(EccTranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;
        let config =
            Rv32WeierstrassConfig::new(vec![SECP256K1_CONFIG.clone(), P256_CONFIG.clone()]);
        air_test(config, openvm_exe);
        Ok(())
    }

    #[test]
    fn test_decompress() -> Result<()> {
        use openvm_ecc_guest::halo2curves::{group::Curve, secp256k1::Secp256k1Affine};

        let elf = build_example_program_at_path_with_features(
            get_programs_dir!(),
            "decompress",
            ["k256"],
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(EccTranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;
        let config = Rv32WeierstrassConfig::new(vec![SECP256K1_CONFIG.clone()]);

        let p = Secp256k1Affine::generator();
        let p = (p + p + p).to_affine();
        println!("decompressed: {:?}", p);
        let coords: Vec<_> = [p.x.to_bytes(), p.y.to_bytes()]
            .concat()
            .into_iter()
            .map(FieldAlgebra::from_canonical_u8)
            .collect();
        air_test_with_min_segments(config, openvm_exe, vec![coords], 1);
        Ok(())
    }

    #[test]
    fn test_ecdsa() -> Result<()> {
        let elf =
            build_example_program_at_path_with_features(get_programs_dir!(), "ecdsa", ["k256"])?;
        let config = SdkVmConfig::builder()
            .system(SystemConfig::default().with_continuations().into())
            .rv32i(Default::default())
            .rv32m(Default::default())
            .io(Default::default())
            .modular(ModularExtension::new(vec![
                SECP256K1_CONFIG.modulus.clone(),
                SECP256K1_CONFIG.scalar.clone(),
            ]))
            .keccak(Default::default())
            .ecc(WeierstrassExtension::new(vec![SECP256K1_CONFIG.clone()]))
            .build();
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(Keccak256TranspilerExtension)
                .with_extension(EccTranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;
        air_test(config, openvm_exe);
        Ok(())
    }
}
