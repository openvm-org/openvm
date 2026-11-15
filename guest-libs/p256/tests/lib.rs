mod guest_tests {
    use ecdsa_config::EcdsaConfig;
    use eyre::Result;
    use openvm_algebra_transpiler::ModularTranspilerExtension;
    use openvm_circuit::{
        arch::instructions::exe::VmExe,
        utils::{air_test, test_system_config},
    };
    use openvm_ecc_circuit::{
        CurveConfig, Rv32WeierstrassBuilder, Rv32WeierstrassConfig, P256_CONFIG,
    };
    use openvm_ecc_transpiler::EccTranspilerExtension;
    use openvm_rv32im_transpiler::{
        Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
    };
    use openvm_sha2_transpiler::Sha2TranspilerExtension;
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

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn test_point_observations() -> Result<()> {
        use std::panic::{catch_unwind, AssertUnwindSafe};

        use openvm_circuit::utils::air_test_with_min_segments;
        use openvm_stark_backend::p3_field::PrimeCharacteristicRing;

        let config = test_rv32weierstrass_config(vec![P256_CONFIG.clone()]);
        let elf = build_example_program_at_path(
            get_programs_dir!("tests/programs"),
            "point_observations",
            &config,
        )?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(EccTranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;
        // Rejection cases rely on executor debug assertions in the fast profile.
        for case in 0..=9 {
            let result = catch_unwind(AssertUnwindSafe(|| {
                air_test_with_min_segments(
                    Rv32WeierstrassBuilder,
                    config.clone(),
                    exe.clone(),
                    vec![vec![F::from_u8(case)]],
                    1,
                )
            }));
            if case == 0 {
                assert!(result.unwrap().is_some());
            } else {
                let Err(panic) = result else {
                    panic!("noncanonical coordinate was accepted");
                };
                let message = panic.downcast_ref::<String>().expect("reduction assertion");
                assert!(
                    message.starts_with('[') && message.contains(" >= ["),
                    "{message}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_add() -> Result<()> {
        let config = test_rv32weierstrass_config(vec![P256_CONFIG.clone()]);
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

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn test_scalar_observations() -> Result<()> {
        use std::panic::{catch_unwind, AssertUnwindSafe};

        use openvm_circuit::utils::air_test_with_min_segments;
        use openvm_stark_backend::p3_field::PrimeCharacteristicRing;

        let config = test_rv32weierstrass_config(vec![P256_CONFIG.clone()]);
        let elf = build_example_program_at_path(
            get_programs_dir!("tests/programs"),
            "scalar_observations",
            &config,
        )?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(EccTranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;
        // Rejection cases rely on executor debug assertions in the fast profile.
        for case in 0..=8 {
            let result = catch_unwind(AssertUnwindSafe(|| {
                air_test_with_min_segments(
                    Rv32WeierstrassBuilder,
                    config.clone(),
                    exe.clone(),
                    vec![vec![F::from_u8(case)]],
                    1,
                )
            }));
            if case == 0 {
                assert!(result.unwrap().is_some());
            } else {
                let Err(panic) = result else {
                    panic!("noncanonical scalar was accepted");
                };
                let message = panic.downcast_ref::<String>().expect("reduction assertion");
                assert!(
                    message.starts_with('[') && message.contains(" >= ["),
                    "{message}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_mul() -> Result<()> {
        let config = test_rv32weierstrass_config(vec![P256_CONFIG.clone()]);
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
        let config = test_rv32weierstrass_config(vec![P256_CONFIG.clone()]);
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
        use openvm_sha2_circuit::{Sha2, Sha2Executor, Sha2ProverExt};
        use serde::{Deserialize, Serialize};
        #[cfg(feature = "cuda")]
        use {
            openvm_circuit::{
                arch::DenseRecordArena,
                openvm_cuda_backend::{BabyBearPoseidon2GpuEngine, GpuBackend},
                system::cuda::SystemChipInventoryGPU,
            },
            openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config,
        };
        #[cfg(not(feature = "cuda"))]
        use {
            openvm_circuit::{
                arch::{MatrixRecordArena, VmField},
                system::SystemChipInventory,
            },
            openvm_cpu_backend::{CpuBackend, CpuDevice},
            openvm_stark_backend::{StarkEngine, StarkProtocolConfig, Val},
        };

        #[derive(Clone, Debug, VmConfig, Serialize, Deserialize)]
        pub struct EcdsaConfig {
            #[config(generics = true)]
            pub weierstrass: Rv32WeierstrassConfig,
            #[extension]
            pub sha2: Sha2,
        }

        impl EcdsaConfig {
            pub fn new(curves: Vec<CurveConfig>) -> Self {
                Self {
                    weierstrass: Rv32WeierstrassConfig::new(curves),
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

        #[cfg(not(feature = "cuda"))]
        impl<E, SC> VmBuilder<E> for EcdsaBuilder
        where
            SC: StarkProtocolConfig,
            E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
            Val<SC>: VmField,
            SC::EF: Ord,
        {
            type VmConfig = EcdsaConfig;
            type SystemChipInventory = SystemChipInventory<SC>;
            type RecordArena = MatrixRecordArena<Val<SC>>;

            fn create_chip_complex(
                &self,
                config: &EcdsaConfig,
                circuit: AirInventory<SC>,
                device_ctx: &openvm_stark_backend::EngineDeviceCtx<E>,
            ) -> Result<
                VmChipComplex<SC, Self::RecordArena, E::PB, Self::SystemChipInventory>,
                ChipInventoryError,
            > {
                let mut chip_complex = VmBuilder::<E>::create_chip_complex(
                    &Rv32WeierstrassBuilder,
                    &config.weierstrass,
                    circuit,
                    device_ctx,
                )?;
                let inventory = &mut chip_complex.inventory;
                VmProverExtension::<E, _, _>::extend_prover(
                    &Sha2ProverExt,
                    &config.sha2,
                    inventory,
                )?;
                Ok(chip_complex)
            }
        }

        #[cfg(feature = "cuda")]
        impl VmBuilder<BabyBearPoseidon2GpuEngine> for EcdsaBuilder {
            type VmConfig = EcdsaConfig;
            type SystemChipInventory = SystemChipInventoryGPU;
            type RecordArena = DenseRecordArena;

            fn create_chip_complex(
                &self,
                config: &EcdsaConfig,
                circuit: AirInventory<BabyBearPoseidon2Config>,
                device_ctx: &openvm_stark_backend::EngineDeviceCtx<BabyBearPoseidon2GpuEngine>,
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
                    VmBuilder::<BabyBearPoseidon2GpuEngine>::create_chip_complex(
                        &Rv32WeierstrassBuilder,
                        &config.weierstrass,
                        circuit,
                        device_ctx,
                    )?;
                let inventory = &mut chip_complex.inventory;
                VmProverExtension::<BabyBearPoseidon2GpuEngine, _, _>::extend_prover(
                    &Sha2ProverExt,
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
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(EccTranspilerExtension)
                .with_extension(ModularTranspilerExtension)
                .with_extension(Sha2TranspilerExtension),
        )?;
        air_test(EcdsaBuilder, config, openvm_exe);
        Ok(())
    }

    #[test]
    fn test_scalar_sqrt() -> Result<()> {
        let config = test_rv32weierstrass_config(vec![P256_CONFIG.clone()]);
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
    use openvm_algebra_guest::IntMod;
    use openvm_ecc_guest::{msm, weierstrass::WeierstrassPoint, Group};
    use p256::{P256Coord, P256Point, P256Scalar};

    #[test]
    fn test_checked_point_bytes() {
        use std::panic::{catch_unwind, AssertUnwindSafe};

        use openvm_ecc_guest::CyclicGroup;

        for x in [false, true] {
            let mut point = P256Point::GENERATOR;
            assert_eq!(point.as_le_bytes(), point.as_le_bytes_unchecked());
            let coord = if x { point.x_mut() } else { point.y_mut() };
            *coord = P256Coord::from_repr(P256Coord::MODULUS);
            assert_eq!(point.as_le_bytes_unchecked().len(), 64);
            assert!(catch_unwind(AssertUnwindSafe(|| point.as_le_bytes())).is_err());
        }
    }

    #[test]
    fn test_checked_byte_access() {
        use std::panic::catch_unwind;

        use elliptic_curve::subtle::{Choice, ConditionallySelectable};

        fn check<T: IntMod + serde::Serialize + ConditionallySelectable>() {
            let raw = T::from_le_bytes_unchecked(T::MODULUS.as_ref());
            assert_eq!(raw.as_le_bytes_unchecked(), T::MODULUS.as_ref());
            let mut expected = T::MODULUS.as_ref().to_vec();
            expected.reverse();
            assert_eq!(raw.to_be_bytes_unchecked().as_ref(), expected);
            assert!(catch_unwind(std::panic::AssertUnwindSafe(|| raw.as_le_bytes())).is_err());
            assert!(catch_unwind(std::panic::AssertUnwindSafe(|| raw.to_be_bytes())).is_err());
            assert!(catch_unwind(std::panic::AssertUnwindSafe(|| raw == T::ZERO)).is_err());
            assert!(catch_unwind(std::panic::AssertUnwindSafe(|| T::ZERO == raw)).is_err());
            assert!(
                catch_unwind(std::panic::AssertUnwindSafe(|| openvm::serde::to_vec(&raw))).is_err()
            );
            let selected = T::conditional_select(&T::ONE, &raw, Choice::from(0));
            assert_eq!(selected.as_le_bytes(), T::ONE.as_le_bytes());
            assert_eq!(
                openvm::serde::to_vec(&selected).unwrap(),
                openvm::serde::to_vec(
                    &<[_; 32]>::try_from(T::ONE.as_le_bytes_unchecked()).unwrap()
                )
                .unwrap()
            );
        }
        check::<P256Scalar>();
        check::<P256Coord>();
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
        let p3 = &p1 + &p2;
        #[allow(clippy::op_ref)]
        let p4 = &p2 + &p2;

        // Add assign and double assign
        let mut sum = unsafe { P256Point::from_xy(x1, y1).unwrap() };
        sum += &p2;
        if sum.x() != p3.x() || sum.y() != p3.y() {
            panic!();
        }
        let mut double = unsafe { P256Point::from_xy(x2, y2).unwrap() };
        double.double_assign();
        if double.x() != p4.x() || double.y() != p4.y() {
            panic!();
        }

        // Ec Mul
        let p1 = unsafe { P256Point::from_xy(x1, y1).unwrap() };
        let scalar = P256Scalar::from_u32(3);
        #[allow(clippy::op_ref)]
        let p2 = &p1.double() + &p1;
        let result = msm(&[scalar], &[p1]);
        if result.x() != p2.x() || result.y() != p2.y() {
            panic!();
        }
    }

    #[test]
    fn test_host_scalar_invert_zero() {
        use elliptic_curve::{ff::Field, ops::Invert};

        let zero = <P256Scalar as Field>::ZERO;
        let one = <P256Scalar as Field>::ONE;
        assert!(bool::from(<P256Scalar as Field>::invert(&zero).is_none()));
        assert!(bool::from(<P256Scalar as Invert>::invert(&zero).is_none()));
        let (is_square, _) = <P256Scalar as Field>::sqrt_ratio(&one, &zero);
        assert!(!bool::from(is_square));

        let two = <P256Scalar as Field>::double(&one);
        let two_inv = <P256Scalar as Field>::invert(&two).unwrap();
        assert_eq!(two * two_inv, one);
    }
}
