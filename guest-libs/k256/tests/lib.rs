mod guest_tests {
    use ecdsa_config::EcdsaConfig;
    use eyre::Result;
    use openvm_algebra_transpiler::ModularTranspilerExtension;
    use openvm_circuit::{
        arch::instructions::exe::VmExe,
        utils::{air_test, test_system_config},
    };
    use openvm_ecc_circuit::{
        CurveConfig, Rv32WeierstrassBuilder, Rv32WeierstrassConfig, SECP256K1_CONFIG,
    };
    use openvm_ecc_transpiler::EccTranspilerExtension;
    use openvm_rv32im_transpiler::{
        Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
    };
    use openvm_sha256_transpiler::Sha256TranspilerExtension;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use openvm_toolchain_tests::{build_example_program_at_path, get_programs_dir};
    use openvm_transpiler::{transpiler::Transpiler, FromElf};

    use crate::guest_tests::ecdsa_config::EcdsaBuilder;

    type F = BabyBear;

    #[cfg(test)]
    fn test_rv32weierstrass_config(curves: Vec<CurveConfig>) -> Rv32WeierstrassConfig {
        let mut config = Rv32WeierstrassConfig::new(curves);
        *config.as_mut() = test_system_config();
        config
    }

    #[test]
    fn test_add() -> Result<()> {
        let config = test_rv32weierstrass_config(vec![SECP256K1_CONFIG.clone()]);
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
        air_test(Rv32WeierstrassBuilder, config, openvm_exe);
        Ok(())
    }

    #[test]
    fn test_mul() -> Result<()> {
        let config = test_rv32weierstrass_config(vec![SECP256K1_CONFIG.clone()]);
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
        air_test(Rv32WeierstrassBuilder, config, openvm_exe);
        Ok(())
    }

    #[test]
    fn test_linear_combination() -> Result<()> {
        let config = test_rv32weierstrass_config(vec![SECP256K1_CONFIG.clone()]);
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
        air_test(Rv32WeierstrassBuilder, config, openvm_exe);
        Ok(())
    }

    // TODO[jpw]: switch to using SDK to avoid this
    mod ecdsa_config {
        use openvm_circuit::{
            arch::{
                AirInventory, ChipInventoryError, InitFileGenerator, SystemConfig, VmBuilder,
                VmChipComplex, VmProverExtension,
            },
            derive::VmConfig,
        };
        use openvm_ecc_circuit::{
            CurveConfig, Rv32WeierstrassBuilder, Rv32WeierstrassConfig,
            Rv32WeierstrassConfigExecutor,
        };
        use openvm_sha256_circuit::{Sha256, Sha256Executor, Sha256ProverExt};
        use serde::{Deserialize, Serialize};
        #[cfg(feature = "cuda")]
        use {
            openvm_circuit::{
                arch::DenseRecordArena,
                openvm_cuda_backend::{
                    engine::GpuBabyBearPoseidon2Engine, prover_backend::GpuBackend,
                },
                system::cuda::SystemChipInventoryGPU,
            },
            openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config,
        };
        #[cfg(not(feature = "cuda"))]
        use {
            openvm_circuit::{arch::MatrixRecordArena, system::SystemChipInventory},
            openvm_stark_backend::{
                config::{StarkGenericConfig, Val},
                engine::StarkEngine,
                p3_field::PrimeField32,
                prover::cpu::{CpuBackend, CpuDevice},
            },
        };

        #[derive(Clone, Debug, VmConfig, Serialize, Deserialize)]
        pub struct EcdsaConfig {
            #[config(generics = true)]
            pub weierstrass: Rv32WeierstrassConfig,
            #[extension]
            pub sha256: Sha256,
        }

        impl EcdsaConfig {
            pub fn new(curves: Vec<CurveConfig>) -> Self {
                Self {
                    weierstrass: Rv32WeierstrassConfig::new(curves),
                    sha256: Default::default(),
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

        #[cfg(not(feature = "cuda"))]
        impl<E, SC> VmBuilder<E> for EcdsaBuilder
        where
            SC: StarkGenericConfig,
            E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
            Val<SC>: PrimeField32,
        {
            type VmConfig = EcdsaConfig;
            type SystemChipInventory = SystemChipInventory<SC>;
            type RecordArena = MatrixRecordArena<Val<SC>>;

            fn create_chip_complex(
                &self,
                config: &EcdsaConfig,
                circuit: AirInventory<SC>,
            ) -> Result<
                VmChipComplex<SC, Self::RecordArena, E::PB, Self::SystemChipInventory>,
                ChipInventoryError,
            > {
                let mut chip_complex = VmBuilder::<E>::create_chip_complex(
                    &Rv32WeierstrassBuilder,
                    &config.weierstrass,
                    circuit,
                )?;
                let inventory = &mut chip_complex.inventory;
                VmProverExtension::<E, _, _>::extend_prover(
                    &Sha256ProverExt,
                    &config.sha256,
                    inventory,
                )?;
                Ok(chip_complex)
            }
        }

        #[cfg(feature = "cuda")]
        impl VmBuilder<GpuBabyBearPoseidon2Engine> for EcdsaBuilder {
            type VmConfig = EcdsaConfig;
            type SystemChipInventory = SystemChipInventoryGPU;
            type RecordArena = DenseRecordArena;

            fn create_chip_complex(
                &self,
                config: &EcdsaConfig,
                circuit: AirInventory<BabyBearPoseidon2Config>,
            ) -> Result<
                VmChipComplex<
                    BabyBearPoseidon2Config,
                    Self::RecordArena,
                    GpuBackend,
                    Self::SystemChipInventory,
                >,
                ChipInventoryError,
            > {
                let mut chip_complex =
                    VmBuilder::<GpuBabyBearPoseidon2Engine>::create_chip_complex(
                        &Rv32WeierstrassBuilder,
                        &config.weierstrass,
                        circuit,
                    )?;
                let inventory = &mut chip_complex.inventory;
                VmProverExtension::<GpuBabyBearPoseidon2Engine, _, _>::extend_prover(
                    &Sha256ProverExt,
                    &config.sha256,
                    inventory,
                )?;
                Ok(chip_complex)
            }
        }
    }

    #[test]
    fn test_ecdsa() -> Result<()> {
        let config = EcdsaConfig::new(vec![SECP256K1_CONFIG.clone()]);

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
        air_test(EcdsaBuilder, config, openvm_exe);
        Ok(())
    }

    #[test]
    fn test_scalar_sqrt() -> Result<()> {
        let config = test_rv32weierstrass_config(vec![SECP256K1_CONFIG.clone()]);
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
        air_test(Rv32WeierstrassBuilder, config, openvm_exe);
        Ok(())
    }
}

mod host_tests {
    use hex_literal::hex;
    use k256::{Scalar as Secp256k1Scalar, Secp256k1Coord, Secp256k1Point};
    use openvm_algebra_guest::IntMod;
    use openvm_ecc_guest::{msm, weierstrass::WeierstrassPoint, Group};

    #[test]
    fn test_host_secp256k1() {
        // Sample points got from https://asecuritysite.com/ecc/ecc_points2 and
        // https://learnmeabitcoin.com/technical/cryptography/elliptic-curve/#add
        let x1 = Secp256k1Coord::from_u32(1);
        let y1 = Secp256k1Coord::from_le_bytes_unchecked(&hex!(
            "EEA7767E580D75BC6FDD7F58D2A84C2614FB22586068DB63B346C6E60AF21842"
        ));
        let x2 = Secp256k1Coord::from_u32(2);
        let y2 = Secp256k1Coord::from_le_bytes_unchecked(&hex!(
            "D1A847A8F879E0AEE32544DA5BA0B3BD1703A1F52867A5601FF6454DD8180499"
        ));
        // This is the sum of (x1, y1) and (x2, y2).
        let x3 = Secp256k1Coord::from_le_bytes_unchecked(&hex!(
            "BE675E31F8AC8200CBCC6B10CECCD6EB93FB07D99BB9E7C99CC9245C862D3AF2"
        ));
        let y3 = Secp256k1Coord::from_le_bytes_unchecked(&hex!(
            "B44573B48FD3416DD256A8C0E1BAD03E88A78BF176778682589B9CB478FC1D79"
        ));
        // This is the double of (x2, y2).
        let x4 = Secp256k1Coord::from_le_bytes_unchecked(&hex!(
            "3BFFFFFF32333333333333333333333333333333333333333333333333333333"
        ));
        let y4 = Secp256k1Coord::from_le_bytes_unchecked(&hex!(
            "AC54ECC4254A4EDCAB10CC557A9811ED1EF7CB8AFDC64820C6803D2C5F481639"
        ));

        let mut p1 = Secp256k1Point::from_xy(x1, y1).unwrap();
        let mut p2 = Secp256k1Point::from_xy(x2, y2).unwrap();

        // Generic add can handle equal or unequal points.
        #[allow(clippy::op_ref)]
        let p3 = &p1 + &p2;
        if p3.x() != &x3 || p3.y() != &y3 {
            panic!();
        }
        #[allow(clippy::op_ref)]
        let p4 = &p2 + &p2;
        if p4.x() != &x4 || p4.y() != &y4 {
            panic!();
        }

        // Add assign and double assign
        p1 += &p2;
        if p1.x() != &x3 || p1.y() != &y3 {
            panic!();
        }
        p2.double_assign();
        if p2.x() != &x4 || p2.y() != &y4 {
            panic!();
        }

        // Ec Mul
        let p1 = Secp256k1Point::from_xy(x1, y1).unwrap();
        let scalar = Secp256k1Scalar::from_u32(12345678);
        // Calculated with https://learnmeabitcoin.com/technical/cryptography/elliptic-curve/#ec-multiply-tool
        let x5 = Secp256k1Coord::from_le_bytes_unchecked(&hex!(
            "194A93387F790803D972AF9C4A40CB89D106A36F58EE2F31DC48A41768216D6D"
        ));
        let y5 = Secp256k1Coord::from_le_bytes_unchecked(&hex!(
            "9E272F746DA7BED171E522610212B6AEEAAFDB2AD9F4B530B8E1B27293B19B2C"
        ));
        let result = msm(&[scalar], &[p1]);
        if result.x() != &x5 || result.y() != &y5 {
            panic!();
        }
    }
}
