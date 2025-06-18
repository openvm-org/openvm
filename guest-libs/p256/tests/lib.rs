mod guest_tests {
    use ecdsa_config::EcdsaConfig;
    use eyre::Result;
    use openvm_algebra_transpiler::ModularTranspilerExtension;
    use openvm_circuit::{arch::instructions::exe::VmExe, utils::air_test};
    use openvm_ecc_circuit::{Rv32WeierstrassConfig, P256_CONFIG};
    use openvm_ecc_transpiler::EccTranspilerExtension;
    use openvm_rv32im_transpiler::{
        Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
    };
    use openvm_sha256_transpiler::Sha256TranspilerExtension;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use openvm_toolchain_tests::{build_example_program_at_path, get_programs_dir};
    use openvm_transpiler::{transpiler::Transpiler, FromElf};

    type F = BabyBear;

    #[test]
    fn test_add() -> Result<()> {
        let config = Rv32WeierstrassConfig::new(vec![P256_CONFIG.clone()]);
        let elf =
            build_example_program_at_path(get_programs_dir!("tests/programs"), "add", &config)?;
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
    fn test_mul() -> Result<()> {
        let config = Rv32WeierstrassConfig::new(vec![P256_CONFIG.clone()]);
        let elf =
            build_example_program_at_path(get_programs_dir!("tests/programs"), "mul", &config)?;
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
    fn test_linear_combination() -> Result<()> {
        let config = Rv32WeierstrassConfig::new(vec![P256_CONFIG.clone()]);
        let elf = build_example_program_at_path(
            get_programs_dir!("tests/programs"),
            "linear_combination",
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

    mod ecdsa_config {
        use eyre::Result;
        use openvm_algebra_circuit::{
            ModularExtension, ModularExtensionExecutor, ModularExtensionPeriphery,
        };
        use openvm_circuit::{
            arch::{InitFileGenerator, SystemConfig},
            derive::VmConfig,
        };
        use openvm_ecc_circuit::{
            CurveConfig, WeierstrassExtension, WeierstrassExtensionExecutor,
            WeierstrassExtensionPeriphery,
        };
        use openvm_rv32im_circuit::{
            Rv32I, Rv32IExecutor, Rv32IPeriphery, Rv32Io, Rv32IoExecutor, Rv32IoPeriphery, Rv32M,
            Rv32MExecutor, Rv32MPeriphery,
        };
        use openvm_sha256_circuit::{Sha256, Sha256Executor, Sha256Periphery};
        use openvm_stark_backend::p3_field::PrimeField32;
        use serde::{Deserialize, Serialize};

        #[derive(Clone, Debug, VmConfig, Serialize, Deserialize)]
        pub struct EcdsaConfig {
            #[system]
            pub system: SystemConfig,
            #[extension]
            pub base: Rv32I,
            #[extension]
            pub mul: Rv32M,
            #[extension]
            pub io: Rv32Io,
            #[extension]
            pub modular: ModularExtension,
            #[extension]
            pub weierstrass: WeierstrassExtension,
            #[extension]
            pub sha256: Sha256,
        }

        impl EcdsaConfig {
            pub fn new(curves: Vec<CurveConfig>) -> Self {
                let primes: Vec<_> = curves
                    .iter()
                    .flat_map(|c| [c.modulus.clone(), c.scalar.clone()])
                    .collect();
                Self {
                    system: SystemConfig::default().with_continuations(),
                    base: Default::default(),
                    mul: Default::default(),
                    io: Default::default(),
                    modular: ModularExtension::new(primes),
                    weierstrass: WeierstrassExtension::new(curves),
                    sha256: Default::default(),
                }
            }
        }

        impl InitFileGenerator for EcdsaConfig {
            fn generate_init_file_contents(&self) -> Option<String> {
                Some(format!(
                    "// This file is automatically generated by cargo openvm. Do not rename or edit.\n{}\n{}\n",
                    self.modular.generate_moduli_init(),
                    self.weierstrass.generate_sw_init()
                ))
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
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(EccTranspilerExtension)
                .with_extension(ModularTranspilerExtension)
                .with_extension(Sha256TranspilerExtension),
        )?;
        air_test(config, openvm_exe);
        Ok(())
    }

    #[test]
    fn test_scalar_sqrt() -> Result<()> {
        let config = Rv32WeierstrassConfig::new(vec![P256_CONFIG.clone()]);
        let elf = build_example_program_at_path(
            get_programs_dir!("tests/programs"),
            "scalar_sqrt",
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
}

mod host_tests {
    use hex_literal::hex;
    use openvm_algebra_guest::IntMod;
    use openvm_ecc_guest::{msm, weierstrass::WeierstrassPoint, Group};
    use p256::{P256Coord, P256Point, P256Scalar};

    #[test]
    fn test_host_p256() {
        // Sample points got from https://asecuritysite.com/ecc/p256p
        let x1 = P256Coord::from_u32(5);
        let y1 = P256Coord::from_le_bytes_unchecked(&hex!(
            "ccfb4832085c4133c5a3d9643c50ca11de7a8199ce3b91fe061858aab9439245"
        ));
        let p1 = P256Point::from_xy(x1, y1).unwrap();
        let x2 = P256Coord::from_u32(6);
        let y2 = P256Coord::from_le_bytes_unchecked(&hex!(
            "cb23828228510d22e9c0e70fb802d1dc47007233e5856946c20a25542c4cb236"
        ));
        let p2 = P256Point::from_xy(x2, y2).unwrap();

        // Generic add can handle equal or unequal points.
        #[allow(clippy::op_ref)]
        let p3 = &p1 + &p2;
        #[allow(clippy::op_ref)]
        let p4 = &p2 + &p2;

        // Add assign and double assign
        let mut sum = P256Point::from_xy(x1, y1).unwrap();
        sum += &p2;
        if sum.x() != p3.x() || sum.y() != p3.y() {
            panic!();
        }
        let mut double = P256Point::from_xy(x2, y2).unwrap();
        double.double_assign();
        if double.x() != p4.x() || double.y() != p4.y() {
            panic!();
        }

        // Ec Mul
        let p1 = P256Point::from_xy(x1, y1).unwrap();
        let scalar = P256Scalar::from_u32(3);
        #[allow(clippy::op_ref)]
        let p2 = &p1.double() + &p1;
        let result = msm(&[scalar], &[p1]);
        if result.x() != p2.x() || result.y() != p2.y() {
            panic!();
        }
    }
}
