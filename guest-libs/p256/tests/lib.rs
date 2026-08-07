mod guest_tests {
    use ecdsa_config::EcdsaConfig;
    use eyre::Result;
    use openvm_algebra_transpiler::ModularTranspilerExtension;
    use openvm_circuit::{
        arch::{instructions::exe::VmExe, Streams},
        utils::{air_test, air_test_impl, test_system_config},
    };
    use openvm_ecc_circuit::{
        CurveConfig, Rv64WeierstrassBuilder, Rv64WeierstrassConfig, P256_CONFIG,
    };
    use openvm_ecc_transpiler::EccTranspilerExtension;
    use openvm_riscv_transpiler::{
        Rv64ITranspilerExtension, Rv64IoTranspilerExtension, Rv64MTranspilerExtension,
    };
    use openvm_sha2_transpiler::Sha2TranspilerExtension;
    use openvm_stark_sdk::{
        config::baby_bear_poseidon2::BabyBearPoseidon2CpuEngine,
        openvm_stark_backend::SystemParams, p3_baby_bear::BabyBear,
    };
    use openvm_toolchain_tests::{build_example_program_at_path, get_programs_dir};
    use openvm_transpiler::{transpiler::Transpiler, FromElf};

    use crate::guest_tests::ecdsa_config::EcdsaBuilder;

    type F = BabyBear;

    #[cfg(test)]
    fn test_rv64weierstrass_config(curves: Vec<CurveConfig>) -> Rv64WeierstrassConfig {
        let mut config = Rv64WeierstrassConfig::new(curves);
        *config.as_mut() = test_system_config();
        config
    }

    #[test]
    fn test_add() -> Result<()> {
        let config = test_rv64weierstrass_config(vec![P256_CONFIG.clone()]);
        let elf =
            build_example_program_at_path(get_programs_dir!("tests/programs"), "add", &config)?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(EccTranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;
        air_test(Rv64WeierstrassBuilder, config, openvm_exe);
        Ok(())
    }

    #[test]
    fn test_mul() -> Result<()> {
        let config = test_rv64weierstrass_config(vec![P256_CONFIG.clone()]);
        let elf =
            build_example_program_at_path(get_programs_dir!("tests/programs"), "mul", &config)?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(EccTranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;
        air_test(Rv64WeierstrassBuilder, config, openvm_exe);
        Ok(())
    }

    #[test]
    fn test_linear_combination() -> Result<()> {
        let config = test_rv64weierstrass_config(vec![P256_CONFIG.clone()]);
        let elf = build_example_program_at_path(
            get_programs_dir!("tests/programs"),
            "linear_combination",
            &config,
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(EccTranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;
        air_test(Rv64WeierstrassBuilder, config, openvm_exe);
        Ok(())
    }

    // TODO[jpw]: switch to using SDK to avoid this
    mod ecdsa_config {
        use openvm_circuit::{
            arch::{
                AirInventory, ChipInventoryError, InitFileGenerator, SystemConfig, VmBuilder,
                VmChipComplex, VmField, VmProverExtension,
            },
            derive::VmConfig,
            system::SystemChipInventory,
        };
        use openvm_cpu_backend::{CpuBackend, CpuDevice};
        use openvm_ecc_circuit::{
            CurveConfig, Rv64WeierstrassConfig, Rv64WeierstrassConfigExecutor,
            Rv64WeierstrassCpuBuilder,
        };
        use openvm_sha2_circuit::{Sha2, Sha2CpuProverExt, Sha2Executor};
        use openvm_stark_backend::{StarkEngine, StarkProtocolConfig, Val};
        use serde::{Deserialize, Serialize};

        #[derive(Clone, Debug, VmConfig, Serialize, Deserialize)]
        pub struct EcdsaConfig {
            #[config]
            pub weierstrass: Rv64WeierstrassConfig,
            #[extension]
            pub sha2: Sha2,
        }

        impl EcdsaConfig {
            pub fn new(curves: Vec<CurveConfig>) -> Self {
                Self {
                    weierstrass: Rv64WeierstrassConfig::new(curves),
                    sha2: Default::default(),
                }
            }
        }

        impl InitFileGenerator for EcdsaConfig {
            fn generate_init_file_contents(&self) -> Option<String> {
                Some(format!(
                    "// This file is automatically generated by cargo openvm. Do not rename or edit.\n{}\n{}\n",
                    self.weierstrass.modular.modular.generate_moduli_init(),
                    self.weierstrass.weierstrass.generate_sw_init()
                ))
            }
        }

        #[derive(Clone)]
        pub struct EcdsaBuilder;

        impl<E, SC> VmBuilder<E> for EcdsaBuilder
        where
            SC: StarkProtocolConfig,
            E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
            Val<SC>: VmField,
            SC::EF: Ord,
        {
            type VmConfig = EcdsaConfig;
            type SystemChipInventory = SystemChipInventory<SC>;

            fn create_chip_complex(
                &self,
                config: &EcdsaConfig,
                circuit: AirInventory<SC>,
                device_ctx: &openvm_stark_backend::EngineDeviceCtx<E>,
            ) -> Result<VmChipComplex<SC, E::PB, Self::SystemChipInventory>, ChipInventoryError>
            {
                let mut chip_complex = VmBuilder::<E>::create_chip_complex(
                    &Rv64WeierstrassCpuBuilder,
                    &config.weierstrass,
                    circuit,
                    device_ctx,
                )?;
                let inventory = &mut chip_complex.inventory;
                VmProverExtension::<E, _>::extend_prover(
                    &Sha2CpuProverExt,
                    &config.sha2,
                    inventory,
                )?;
                Ok(chip_complex)
            }
        }
    }

    #[test]
    fn test_ecdsa() -> Result<()> {
        let config = EcdsaConfig::new(vec![P256_CONFIG.clone()]);

        let elf =
            build_example_program_at_path(get_programs_dir!("tests/programs"), "ecdsa", &config)?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(EccTranspilerExtension)
                .with_extension(ModularTranspilerExtension)
                .with_extension(Sha2TranspilerExtension),
        )?;
        let debug = std::env::var("OPENVM_SKIP_DEBUG") != Ok("1".to_string());
        air_test_impl::<BabyBearPoseidon2CpuEngine, _>(
            SystemParams::new_for_testing(22),
            EcdsaBuilder,
            config,
            openvm_exe,
            Streams::default(),
            1,
            debug,
        )
        .unwrap();
        Ok(())
    }

    #[test]
    fn test_scalar_sqrt() -> Result<()> {
        let config = test_rv64weierstrass_config(vec![P256_CONFIG.clone()]);
        let elf = build_example_program_at_path(
            get_programs_dir!("tests/programs"),
            "scalar_sqrt",
            &config,
        )?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(EccTranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;
        air_test(Rv64WeierstrassBuilder, config, openvm_exe);
        Ok(())
    }
}

mod host_tests {
    use elliptic_curve::subtle::ConstantTimeEq;
    use hex_literal::hex;
    use openvm_algebra_guest::IntMod;
    use openvm_ecc_guest::{
        msm,
        weierstrass::{CachedMulTable, IntrinsicCurve, WeierstrassPoint},
        CyclicGroup, Group,
    };
    #[cfg(feature = "ecdsa-core")]
    use p256::ecdsa::VerifyingKey;
    use p256::{NistP256, P256Coord, P256Point, P256Scalar};

    #[test]
    fn test_projective_coordinate_contracts() {
        let generator = P256Point::GENERATOR;
        let expected = generator.into_affine_coords().unwrap();
        assert_eq!(expected.0, generator.x().clone());
        assert_eq!(expected.1, generator.y().clone());

        let scale = P256Coord::from_u32(7);
        let scaled = unsafe {
            P256Point::from_xyz_unchecked(
                generator.x() * &scale,
                generator.y() * &scale,
                generator.z() * &scale,
            )
        };
        assert_eq!(scaled.into_affine_coords(), Some(expected));
        assert!(<P256Point as WeierstrassPoint>::IDENTITY
            .into_affine_coords()
            .is_none());

        let malformed_identity = unsafe {
            P256Point::from_xyz_unchecked(P256Coord::ZERO, P256Coord::ZERO, P256Coord::ZERO)
        };
        assert_ne!(malformed_identity, generator);
        assert!(!bool::from(malformed_identity.ct_eq(&generator)));

        let encoded = serde_json::to_vec(&malformed_identity).unwrap();
        let decoded: P256Point = serde_json::from_slice(&encoded).unwrap();
        assert!(Group::is_identity(&decoded));
        assert_ne!(decoded, generator);
        assert_eq!(decoded, <P256Point as WeierstrassPoint>::IDENTITY);
        assert_eq!(decoded + generator, generator);

        #[cfg(feature = "ecdsa-core")]
        {
            let verifying_key = VerifyingKey::from_affine(scaled).unwrap();
            assert_eq!(verifying_key.as_affine().z(), &P256Coord::ONE);
        }
    }

    #[test]
    fn test_cached_mul_table_matches_msm() {
        let bases = [P256Point::GENERATOR];
        let table = CachedMulTable::<NistP256>::new_with_prime_order(&bases, 4);
        for scalar in [0, 1, 2, 3, 7, 15, 16, 255] {
            let scalar = P256Scalar::from_u32(scalar);
            assert_eq!(table.windowed_mul(&[scalar]), msm(&[scalar], &bases));
        }
    }

    #[test]
    fn test_fixed_generator_lincomb_matches_msm() {
        let point = P256Point::GENERATOR.double();
        let wide = P256Scalar::from_le_bytes_unchecked(&hex!(
            "efcdab896745230fefcdab896745230fefcdab896745230fefcdab896745230f"
        ));
        let scalars = [
            (P256Scalar::from_u32(0), P256Scalar::from_u32(0)),
            (P256Scalar::from_u32(1), P256Scalar::from_u32(1)),
            (P256Scalar::from_u32(7), P256Scalar::from_u32(15)),
            (P256Scalar::from_u32(255), P256Scalar::from_u32(65_537)),
            (wide, P256Scalar::from_u32(3)),
        ];
        for (generator_scalar, point_scalar) in scalars {
            assert_eq!(
                <NistP256 as IntrinsicCurve>::lincomb_generator(
                    &generator_scalar,
                    &point_scalar,
                    &point,
                ),
                msm(
                    &[generator_scalar, point_scalar],
                    &[P256Point::GENERATOR, point]
                )
            );
            assert_eq!(
                <NistP256 as IntrinsicCurve>::lincomb_neg_generator(
                    &generator_scalar,
                    &point_scalar,
                    &point,
                ),
                msm(
                    &[generator_scalar, point_scalar],
                    &[P256Point::NEG_GENERATOR, point]
                )
            );
        }
    }

    #[test]
    fn test_host_p256() {
        // Sample points got from https://asecuritysite.com/ecc/p256p
        let x1 = P256Coord::from_u32(5);
        let y1 = P256Coord::from_le_bytes_unchecked(&hex!(
            "ccfb4832085c4133c5a3d9643c50ca11de7a8199ce3b91fe061858aab9439245"
        ));
        let p1 = unsafe { P256Point::from_xy(x1, y1).unwrap() };
        let x2 = P256Coord::from_u32(6);
        let y2 = P256Coord::from_le_bytes_unchecked(&hex!(
            "cb23828228510d22e9c0e70fb802d1dc47007233e5856946c20a25542c4cb236"
        ));
        let p2 = unsafe { P256Point::from_xy(x2, y2).unwrap() };

        // Generic add can handle equal or unequal points.
        #[allow(clippy::op_ref)]
        let p3 = (&p1 + &p2).normalize();
        #[allow(clippy::op_ref)]
        let p4 = (&p2 + &p2).normalize();

        // Add assign and double assign
        let mut sum = unsafe { P256Point::from_xy(x1, y1).unwrap() };
        sum += &p2;
        let sum = sum.normalize();
        if sum.x() != p3.x() || sum.y() != p3.y() {
            panic!();
        }
        let mut double = unsafe { P256Point::from_xy(x2, y2).unwrap() };
        double.double_assign();
        let double = double.normalize();
        if double.x() != p4.x() || double.y() != p4.y() {
            panic!();
        }

        // Ec Mul
        let p1 = unsafe { P256Point::from_xy(x1, y1).unwrap() };
        let scalar = P256Scalar::from_u32(3);
        #[allow(clippy::op_ref)]
        let p2 = (&p1.double() + &p1).normalize();
        let result = msm(&[scalar], &[p1]).normalize();
        if result.x() != p2.x() || result.y() != p2.y() {
            panic!();
        }
    }
}
