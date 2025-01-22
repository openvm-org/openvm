#[cfg(test)]
mod tests {
    use core::str::FromStr;

    use eyre::Result;
    use hex_literal::hex;
    use num_bigint::BigUint;
    use openvm_algebra_circuit::ModularExtension;
    use openvm_algebra_transpiler::ModularTranspilerExtension;
    use openvm_circuit::{
        arch::{instructions::exe::VmExe, SystemConfig},
        utils::{air_test, air_test_with_min_segments},
    };
    use openvm_ecc_circuit::{
        CurveCoeffs, CurveConfig, EccExtension, Rv32EccConfig, TeCurveConfig, P256_CONFIG,
        SECP256K1_CONFIG,
    };
    use openvm_ecc_transpiler::EccTranspilerExtension;
    use openvm_keccak256_transpiler::Keccak256TranspilerExtension;
    use openvm_rv32im_transpiler::{
        Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
    };
    use openvm_sdk::config::SdkVmConfig;
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::{openvm_stark_backend, p3_baby_bear::BabyBear};
    use openvm_toolchain_tests::{
        build_example_program_at_path_with_features, get_programs_dir, NoInitFile,
    };
    use openvm_transpiler::{transpiler::Transpiler, FromElf};
    type F = BabyBear;

    #[test]
    fn test_ec() -> Result<()> {
        let config = Rv32EccConfig::new(vec![SECP256K1_CONFIG.clone()]);
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!(),
            "ec",
            ["k256"],
            &config,
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
        air_test(config, openvm_exe);
        Ok(())
    }

    #[test]
    fn test_ec_nonzero_a() -> Result<()> {
        let config = Rv32EccConfig::new(vec![P256_CONFIG.clone()]);
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!(),
            "ec_nonzero_a",
            ["p256"],
            &config,
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
        air_test(config, openvm_exe);
        Ok(())
    }

    #[test]
    fn test_ec_two_curves() -> Result<()> {
        let config = Rv32EccConfig::new(vec![SECP256K1_CONFIG.clone(), P256_CONFIG.clone()]);
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!(),
            "ec_two_curves",
            ["k256", "p256"],
            &config,
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
        air_test(config, openvm_exe);
        Ok(())
    }

    #[test]
    fn test_decompress() -> Result<()> {
        use halo2curves_axiom::{group::Curve, secp256k1::Secp256k1Affine};

        let config =
            Rv32EccConfig::new(vec![SECP256K1_CONFIG.clone(),
                CurveConfig {
                    struct_name: "CurvePoint5mod8".to_string(),
                    modulus: BigUint::from_str("115792089237316195423570985008687907853269984665640564039457584007913129639501")
                        .unwrap(),
                    // unused, set to 10e9 + 7
                    scalar: BigUint::from_str("1000000007")
                        .unwrap(),
                    coeffs: CurveCoeffs::SwCurve(SwCurveConfig {
                        a: BigUint::ZERO,
                        b: BigUint::from_str("3").unwrap(),
                    }),
                },
                CurveConfig {
                    struct_name: "CurvePoint1mod4".to_string(),
                    modulus: BigUint::from_radix_be(&hex!("ffffffffffffffffffffffffffffffff000000000000000000000001"), 256)
                        .unwrap(),
                    scalar: BigUint::from_radix_be(&hex!("ffffffffffffffffffffffffffff16a2e0b8f03e13dd29455c5c2a3d"), 256)
                        .unwrap(),
                    coeffs: CurveCoeffs::SwCurve(SwCurveConfig {
                        a: BigUint::from_radix_be(&hex!("fffffffffffffffffffffffffffffffefffffffffffffffffffffffe"), 256)
                            .unwrap(),
                        b: BigUint::from_radix_be(&hex!("b4050a850c04b3abf54132565044b0b7d7bfd8ba270b39432355ffb4"), 256)
                            .unwrap(),
                    }),
                },
            ]);

        let elf = build_example_program_at_path_with_features(
            get_programs_dir!(),
            "decompress",
            ["k256"],
            &config,
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

        let p = Secp256k1Affine::generator();
        let p = (p + p + p).to_affine();
        println!("decompressed: {:?}", p);
        let q_x: [u8; 32] =
            hex!("0100000000000000000000000000000000000000000000000000000000000000");
        let q_y: [u8; 32] =
            hex!("0200000000000000000000000000000000000000000000000000000000000000");
        let r_x: [u8; 32] =
            hex!("211D5C11D68032342211C256D3C1034AB99013327FBFB46BBD0C0EB700000000");
        let r_y: [u8; 32] =
            hex!("347E00859981D5446447075AA07543CDE6DF224CFB23F7B5886337BD00000000");

        let coords = [p.x.to_bytes(), p.y.to_bytes(), q_x, q_y, r_x, r_y]
            .concat()
            .into_iter()
            .map(FieldAlgebra::from_canonical_u8)
            .collect();
        air_test_with_min_segments(config, openvm_exe, vec![coords], 1);
        Ok(())
    }

    #[test]
    fn test_ecdsa() -> Result<()> {
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
            .ecc(EccExtension::new(vec![SECP256K1_CONFIG.clone()]))
            .build();

        let elf = build_example_program_at_path_with_features(
            get_programs_dir!(),
            "ecdsa",
            ["k256"],
            &config,
        )?;
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

    #[test]
    fn test_edwards_ec() -> Result<()> {
        let elf = build_example_program_at_path_with_features::<&str>(
            get_programs_dir!(),
            "edwards_ec",
            [],
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
            Rv32EccConfig::new(vec![CurveConfig {
            modulus: BigUint::from_str(
                "57896044618658097711785492504343953926634992332820282019728792003956564819949",
            ).unwrap(),
            scalar: BigUint::from_str(
                "7237005577332262213973186563042994240857116359379907606001950938285454250989",
            ).unwrap(),
            coeffs: CurveCoeffs::TeCurve(TeCurveConfig {
                a: BigUint::from_str(
                    "57896044618658097711785492504343953926634992332820282019728792003956564819948",
                ).unwrap(),
                d: BigUint::from_str(
                    "37095705934669439343138083508754565189542113879843219016388785533085940283555",
                ).unwrap(),
            }),
        }]);
        air_test(config, openvm_exe);
        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_invalid_setup() {
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!(),
            "invalid_setup",
            ["k256", "p256"],
            &NoInitFile, // don't use build script since we are testing invalid setup
        )
        .unwrap();
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(EccTranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )
        .unwrap();
        let config =
            Rv32WeierstrassConfig::new(vec![SECP256K1_CONFIG.clone(), P256_CONFIG.clone()]);
        air_test(config, openvm_exe);
    }
}
