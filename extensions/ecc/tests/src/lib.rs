#[cfg(test)]
mod tests {

    use derive_more::derive::From;
    use eyre::Result;
    use num_bigint_dig::BigUint;
    use openvm_algebra_circuit::{
        ModularExtension, ModularExtensionExecutor, ModularExtensionPeriphery,
    };
    use openvm_algebra_transpiler::ModularTranspilerExtension;
    use openvm_circuit::{
        arch::{
            instructions::exe::VmExe, SystemConfig, SystemExecutor, SystemPeriphery, VmChipComplex,
            VmConfig, VmInventoryError,
        },
        derive::{AnyEnum, InstructionExecutor, VmConfig},
        utils::new_air_test_with_min_segments,
    };
    use openvm_circuit_primitives_derive::{Chip, ChipUsageGetter};
    use openvm_ecc_circuit::{
        CurveConfig, Rv32WeierstrassConfig, WeierstrassExtension, WeierstrassExtensionExecutor,
        WeierstrassExtensionPeriphery, SECP256K1_CONFIG,
    };
    use openvm_ecc_transpiler::EccTranspilerExtension;
    use openvm_keccak256_circuit::{Keccak256, Keccak256Executor, Keccak256Periphery};
    use openvm_keccak256_transpiler::Keccak256TranspilerExtension;
    use openvm_rv32im_circuit::{
        Rv32I, Rv32IExecutor, Rv32IPeriphery, Rv32Io, Rv32IoExecutor, Rv32IoPeriphery, Rv32M,
        Rv32MExecutor, Rv32MPeriphery,
    };
    use openvm_rv32im_transpiler::{
        Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
    };
    use openvm_stark_backend::p3_field::{AbstractField, PrimeField32};
    use openvm_stark_sdk::{openvm_stark_backend, p3_baby_bear::BabyBear};
    use openvm_toolchain_tests::{build_example_program_at_path_with_features, get_programs_dir};
    use openvm_transpiler::{transpiler::Transpiler, FromElf};
    use serde::{Deserialize, Serialize};
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
        new_air_test_with_min_segments(config, openvm_exe, vec![], 1, false);
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
            .map(AbstractField::from_canonical_u8)
            .collect();
        new_air_test_with_min_segments(config, openvm_exe, vec![coords], 1, false);
        Ok(())
    }

    #[derive(Clone, Debug, VmConfig, Serialize, Deserialize)]
    pub struct Rv32ModularKeccak256Config {
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
        pub keccak: Keccak256,
        #[extension]
        pub weierstrass: WeierstrassExtension,
    }

    impl Rv32ModularKeccak256Config {
        pub fn new(curves: Vec<CurveConfig>) -> Self {
            let primes: Vec<BigUint> = curves
                .iter()
                .flat_map(|c| [c.modulus.clone(), c.scalar.clone()])
                .collect();
            Self {
                system: SystemConfig::default().with_continuations(),
                base: Default::default(),
                mul: Default::default(),
                io: Default::default(),
                modular: ModularExtension::new(primes),
                keccak: Default::default(),
                weierstrass: WeierstrassExtension::new(curves),
            }
        }
    }

    #[test]
    fn test_ecdsa() -> Result<()> {
        let elf =
            build_example_program_at_path_with_features(get_programs_dir!(), "ecdsa", ["k256"])?;
        let config = Rv32ModularKeccak256Config::new(vec![SECP256K1_CONFIG.clone()]);

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
        new_air_test_with_min_segments(config, openvm_exe, vec![], 1, true);
        Ok(())
    }
}
