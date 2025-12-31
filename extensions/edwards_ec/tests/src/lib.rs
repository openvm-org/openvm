#[cfg(test)]
mod tests {
    use eyre::Result;
    use num_bigint::BigUint;
    use num_traits::{cast::FromPrimitive, identities::One};
    use openvm_algebra_transpiler::ModularTranspilerExtension;
    use openvm_circuit::{
        arch::instructions::exe::VmExe,
        utils::{air_test, air_test_with_min_segments, test_system_config},
    };
    use openvm_edwards_circuit::{
        CurveConfig, Rv32EdwardsBuilder, Rv32EdwardsConfig, ED25519_CONFIG,
    };
    use openvm_rv32im_transpiler::{
        Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
    };
    use openvm_sdk::config::{AppConfig, SdkVmBuilder, SdkVmConfig};
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::{openvm_stark_backend, p3_baby_bear::BabyBear};
    use openvm_edwards_transpiler::EdwardsTranspilerExtension;
    use openvm_toolchain_tests::{
        build_example_program_at_path_with_features, get_programs_dir, NoInitFile,
    };
    use openvm_transpiler::{transpiler::Transpiler, FromElf};

    type F = BabyBear;

    #[cfg(test)]
    fn test_rv32edwards_config(curves: Vec<CurveConfig>) -> Rv32EdwardsConfig {
        let mut config = Rv32EdwardsConfig::new(curves);
        *config.as_mut() = test_system_config();
        config
    }

    #[test]
    fn test_decompress() -> Result<()> {
        use halo2curves_axiom::{ed25519::Ed25519Affine, group::Curve};

        let config = test_rv32edwards_config(vec![ED25519_CONFIG.clone()]);

        let elf = build_example_program_at_path_with_features(
            get_programs_dir!(),
            "decompress",
            ["ed25519"],
            &config,
        )?;

        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(EdwardsTranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;

        let s = Ed25519Affine::generator();
        let s = (s + s + s).to_affine();

        let coords = [s.x.to_bytes(), s.y.to_bytes()]
            .concat()
            .into_iter()
            .map(FieldAlgebra::from_canonical_u8)
            .collect();

        air_test_with_min_segments(Rv32EdwardsBuilder, config, openvm_exe, vec![coords], 1);
        Ok(())
    }

    #[test]
    fn test_edwards_ec() -> Result<()> {
        let config = toml::from_str::<AppConfig<SdkVmConfig>>(include_str!(
            "../programs/openvm_ed25519.toml"
        ))?
        .app_vm_config;
        let elf = build_example_program_at_path_with_features::<&str>(
            get_programs_dir!(),
            "edwards_ec",
            ["ed25519"],
            &config,
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(EdwardsTranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;
        air_test(SdkVmBuilder, config, openvm_exe);
        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_invalid_setup() {
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!(),
            "invalid_setup",
            ["ed25519"],
            &NoInitFile, // don't use build script since we are testing invalid setup
        )
        .unwrap();
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(EdwardsTranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )
        .unwrap();
        let config = test_rv32edwards_config(vec![
            ED25519_CONFIG.clone(),
            CurveConfig::new(
                String::from("SampleCurvePoint"),
                ED25519_CONFIG.modulus.clone(),
                BigUint::one(), // dummy value, since order is unknown and unused
                BigUint::from_u32(3).unwrap(),
                BigUint::from_u32(2).unwrap(),
            ),
        ]);
        air_test(Rv32EdwardsBuilder, config, openvm_exe);
    }
}
