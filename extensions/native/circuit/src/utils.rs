pub(crate) const CASTF_MAX_BITS: usize = 30;

pub(crate) const fn const_max(a: usize, b: usize) -> usize {
    [a, b][(a < b) as usize]
}

/// Testing framework
#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils {
    use std::array;

    use openvm_circuit::{
        arch::{
            execution_mode::Segment,
            testing::{memory::gen_pointer, TestBuilder},
            PreflightExecutionOutput, PreflightExecutor, Streams, VirtualMachine,
            VirtualMachineError, VmBuilder, VmExecutionConfig, VmState,
        },
        utils::test_system_config_without_continuations,
    };
    use openvm_instructions::{
        exe::VmExe,
        program::Program,
        riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
    };
    use openvm_native_compiler::conversion::AS;
    use openvm_stark_backend::{
        config::Domain, p3_commit::PolynomialSpace, p3_field::PrimeField32,
    };
    use openvm_stark_sdk::{
        config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, setup_tracing, FriParameters},
        engine::StarkFriEngine,
        p3_baby_bear::BabyBear,
    };
    use rand::{distributions::Standard, prelude::Distribution, rngs::StdRng, Rng};

    use crate::{NativeConfig, NativeCpuBuilder, Rv32WithKernelsConfig};

    // If immediate, returns (value, AS::Immediate). Otherwise, writes to native memory and returns
    // (ptr, AS::Native). If is_imm is None, randomizes it.
    pub fn write_native_or_imm<F: PrimeField32>(
        tester: &mut impl TestBuilder<F>,
        rng: &mut StdRng,
        value: F,
        is_imm: Option<bool>,
    ) -> (F, usize) {
        let is_imm = is_imm.unwrap_or(rng.gen_bool(0.5));
        if is_imm {
            (value, AS::Immediate as usize)
        } else {
            let ptr = gen_pointer(rng, 1);
            tester.write::<1>(AS::Native as usize, ptr, [value]);
            (F::from_canonical_usize(ptr), AS::Native as usize)
        }
    }

    // Writes value to native memory and returns a pointer to the first element together with the
    // value If `value` is None, randomizes it.
    pub fn write_native_array<F: PrimeField32, const N: usize>(
        tester: &mut impl TestBuilder<F>,
        rng: &mut StdRng,
        value: Option<[F; N]>,
    ) -> ([F; N], usize)
    where
        Standard: Distribution<F>, // Needed for `rng.gen`
    {
        let value = value.unwrap_or(array::from_fn(|_| rng.gen()));
        let ptr = gen_pointer(rng, N);
        tester.write::<N>(AS::Native as usize, ptr, value);
        (value, ptr)
    }

    // Besides taking in system_config, this also returns Result and the full
    // (PreflightExecutionOutput, VirtualMachine) for more advanced testing needs.
    #[allow(clippy::type_complexity)]
    pub fn execute_program_with_config<E, VB>(
        program: Program<BabyBear>,
        input_stream: impl Into<Streams<BabyBear>>,
        builder: VB,
        config: VB::VmConfig,
    ) -> Result<
        (
            PreflightExecutionOutput<BabyBear, <VB as VmBuilder<E>>::RecordArena>,
            VirtualMachine<E, VB>,
        ),
        VirtualMachineError,
    >
    where
        E: StarkFriEngine,
        Domain<E::SC>: PolynomialSpace<Val = BabyBear>,
        VB: VmBuilder<E, VmConfig = NativeConfig>,
        <VB::VmConfig as VmExecutionConfig<BabyBear>>::Executor:
            PreflightExecutor<BabyBear, VB::RecordArena>,
    {
        setup_tracing();
        assert!(!config.as_ref().continuation_enabled);
        let input = input_stream.into();

        let engine = E::new(FriParameters::new_for_testing(1));
        let exe = VmExe::new(program);
        let (vm, _) = VirtualMachine::new_with_keygen(engine, builder, config)?;
        let ctx = vm.build_metered_ctx(&exe);
        let (mut segments, _) = vm
            .metered_interpreter(&exe)?
            .execute_metered(input.clone(), ctx)?;
        assert_eq!(segments.len(), 1, "test only supports one segment");
        let Segment {
            instret_start,
            num_insns,
            trace_heights,
        } = segments.pop().unwrap();
        assert_eq!(instret_start, 0);
        let state = vm.create_initial_state(&exe, input);
        let mut preflight_interpreter = vm.preflight_interpreter(&exe)?;
        let output =
            vm.execute_preflight(&mut preflight_interpreter, state, None, &trace_heights)?;
        assert_eq!(
            output.to_state.instret(),
            num_insns,
            "metered execution insn count doesn't match preflight execution"
        );
        Ok((output, vm))
    }

    pub fn execute_program(
        program: Program<BabyBear>,
        input_stream: impl Into<Streams<BabyBear>>,
    ) -> VmState<BabyBear> {
        let mut config = test_native_config();
        config.system.num_public_values = 4;
        // we set max segment len large so it doesn't segment
        let (output, _) = execute_program_with_config::<BabyBearPoseidon2Engine, _>(
            program,
            input_stream,
            NativeCpuBuilder,
            config,
        )
        .unwrap();
        output.to_state
    }

    pub fn test_native_config() -> NativeConfig {
        let mut system = test_system_config_without_continuations();
        system.memory_config.addr_spaces[RV32_REGISTER_AS as usize].num_cells = 0;
        system.memory_config.addr_spaces[RV32_MEMORY_AS as usize].num_cells = 0;
        NativeConfig {
            system,
            native: Default::default(),
        }
    }

    pub fn test_native_continuations_config() -> NativeConfig {
        NativeConfig {
            system: test_system_config_without_continuations().with_continuations(),
            native: Default::default(),
        }
    }

    pub fn test_rv32_with_kernels_config() -> Rv32WithKernelsConfig {
        Rv32WithKernelsConfig {
            system: test_system_config_without_continuations().with_continuations(),
            ..Default::default()
        }
    }
}
