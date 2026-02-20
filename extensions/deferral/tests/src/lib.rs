#[cfg(test)]
#[cfg(not(feature = "cuda"))]
mod tests {
    use std::sync::Arc;

    use eyre::Result;
    use openvm_circuit::{
        arch::{
            deferral::{DeferralResult, DeferralState},
            Streams,
        },
        utils::{air_test_with_min_segments, test_system_config},
    };
    use openvm_deferral_circuit::{
        DeferralCpuBuilder, DeferralExtension, DeferralFn, Rv32DeferralConfig,
    };
    use openvm_deferral_transpiler::DeferralTranspilerExtension;
    use openvm_instructions::{exe::VmExe, NATIVE_AS};
    use openvm_rv32im_circuit::{Rv32I, Rv32Io, Rv32M};
    use openvm_rv32im_transpiler::{
        Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
    };
    use openvm_stark_sdk::{
        config::baby_bear_poseidon2::DIGEST_SIZE,
        openvm_stark_backend::p3_field::PrimeCharacteristicRing, p3_baby_bear::BabyBear,
    };
    use openvm_toolchain_tests::{build_example_program_at_path, get_programs_dir};
    use openvm_transpiler::{transpiler::Transpiler, FromElf};

    type F = BabyBear;

    const INPUT_COMMIT_0: [u8; 32] = [0x11; 32];
    const INPUT_COMMIT_1: [u8; 32] = [0x22; 32];
    const INPUT_COMMIT_2: [u8; 32] = [0x33; 32];

    const INPUT_RAW_0: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    const INPUT_RAW_1: [u8; 8] = [8, 7, 6, 5, 4, 3, 2, 1];
    const INPUT_RAW_2: [u8; 8] = [9, 9, 9, 9, 9, 9, 9, 9];

    fn make_config(num_deferrals: usize) -> Rv32DeferralConfig {
        let mut system = test_system_config();
        system.memory_config.addr_spaces[NATIVE_AS as usize].num_cells = 1 << 25;
        Rv32DeferralConfig {
            system,
            rv32i: Rv32I,
            rv32m: Rv32M::default(),
            io: Rv32Io,
            deferral: make_deferral_extension(num_deferrals),
        }
    }

    fn make_deferral_extension(num_deferrals: usize) -> DeferralExtension {
        let fns: Vec<_> = (0..num_deferrals)
            .map(|idx| {
                Arc::new(DeferralFn::new(move |input_raw| {
                    let mut prefix_sum = 0u16;
                    input_raw
                        .iter()
                        .map(|&byte| {
                            prefix_sum += byte as u16;
                            (prefix_sum + idx as u16) as u8
                        })
                        .collect()
                }))
            })
            .collect();

        let commits = (0..num_deferrals)
            .map(|_| [F::ZERO; DIGEST_SIZE])
            .collect::<Vec<_>>();
        DeferralExtension::new::<F>(fns, &commits)
    }

    fn run_test(config: Rv32DeferralConfig, example_name: &str, streams: Streams<F>) -> Result<()> {
        let elf = build_example_program_at_path(get_programs_dir!(), example_name, &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(DeferralTranspilerExtension),
        )?;
        air_test_with_min_segments(DeferralCpuBuilder, config, exe, streams, 1).unwrap();
        Ok(())
    }

    #[test]
    fn test_deferral_single() -> Result<()> {
        let mut state = DeferralState::new(Vec::<DeferralResult>::new());
        state.store_input(INPUT_COMMIT_0.to_vec(), INPUT_RAW_0.to_vec());
        state.store_input(INPUT_COMMIT_1.to_vec(), INPUT_RAW_1.to_vec());
        state.store_input(INPUT_COMMIT_2.to_vec(), INPUT_RAW_2.to_vec());

        let streams = Streams {
            deferrals: vec![state],
            ..Default::default()
        };
        run_test(make_config(1), "single", streams)
    }

    #[test]
    fn test_deferral_multiple() -> Result<()> {
        let mut state0 = DeferralState::new(Vec::<DeferralResult>::new());
        state0.store_input(INPUT_COMMIT_0.to_vec(), INPUT_RAW_0.to_vec());
        state0.store_input(INPUT_COMMIT_1.to_vec(), INPUT_RAW_1.to_vec());
        state0.store_input(INPUT_COMMIT_2.to_vec(), INPUT_RAW_2.to_vec());

        let mut state1 = DeferralState::new(Vec::<DeferralResult>::new());
        state1.store_input(INPUT_COMMIT_0.to_vec(), INPUT_RAW_0.to_vec());

        let streams = Streams {
            deferrals: vec![state0, state1],
            ..Default::default()
        };
        run_test(make_config(2), "multiple", streams)
    }
}
