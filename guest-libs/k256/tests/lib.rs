#[cfg(test)]
mod tests {
    use ecdsa_config::EcdsaConfig;
    use eyre::Result;
    use openvm_algebra_transpiler::ModularTranspilerExtension;
    use openvm_circuit::{arch::instructions::exe::VmExe, utils::air_test};
    use openvm_ecc_circuit::{Rv32WeierstrassConfig, SECP256K1_CONFIG};
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
        let config = Rv32WeierstrassConfig::new(vec![SECP256K1_CONFIG.clone()]);
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
        let config = Rv32WeierstrassConfig::new(vec![SECP256K1_CONFIG.clone()]);
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
        let config = Rv32WeierstrassConfig::new(vec![SECP256K1_CONFIG.clone()]);
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
        air_test(config, openvm_exe);
        Ok(())
    }

    #[test]
    fn test_sjcalar_sqrt() -> Result<()> {
        let config = Rv32WeierstrassConfig::new(vec![SECP256K1_CONFIG.clone()]);
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
