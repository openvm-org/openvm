#[cfg(test)]
mod tests {
    use eyre::Result;
    use openvm_algebra_transpiler::ModularTranspilerExtension;
    use openvm_circuit::{
        arch::VmExecutor,
        utils::{air_test, test_system_config},
    };
    use openvm_ecc_circuit::{Rv64WeierstrassBuilder, Rv64WeierstrassConfig, P256_CONFIG};
    use openvm_ecc_transpiler::EccTranspilerExtension;
    use openvm_instructions::exe::VmExe;
    use openvm_riscv_transpiler::{
        Rv64ITranspilerExtension, Rv64IoTranspilerExtension, Rv64MTranspilerExtension,
    };
    use openvm_sdk::StdIn;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use openvm_toolchain_tests::{build_example_program_at_path, get_programs_dir};
    use openvm_transpiler::{transpiler::Transpiler, FromElf};

    type F = BabyBear;

    fn test_config() -> Rv64WeierstrassConfig {
        let mut config = Rv64WeierstrassConfig::new(vec![P256_CONFIG.clone()]);
        *config.as_mut() = test_system_config();
        config
    }

    fn test_zkvm_secp256r1_base(prove: bool) -> Result<()> {
        let config = test_config();
        let elf =
            build_example_program_at_path(get_programs_dir!("tests/programs"), "p256", &config)?;
        let openvm_exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv64ITranspilerExtension)
                .with_extension(Rv64MTranspilerExtension)
                .with_extension(Rv64IoTranspilerExtension)
                .with_extension(EccTranspilerExtension)
                .with_extension(ModularTranspilerExtension),
        )?;

        if prove {
            air_test(Rv64WeierstrassBuilder, config, openvm_exe);
        } else {
            let executor = VmExecutor::new(config)?;
            let instance = executor.instance(&openvm_exe)?;
            instance.execute(StdIn::default())?;
        }
        Ok(())
    }

    #[test]
    fn test_zkvm_secp256r1_run() -> Result<()> {
        test_zkvm_secp256r1_base(false)
    }

    #[test]
    #[ignore = "proving on CPU is slow"]
    fn test_zkvm_secp256r1_prove() -> Result<()> {
        test_zkvm_secp256r1_base(true)
    }
}
