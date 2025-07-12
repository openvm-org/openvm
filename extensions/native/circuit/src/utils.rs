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
            execution_mode::metered::Segment,
            testing::{memory::gen_pointer, VmChipTestBuilder},
            Streams, SystemConfig, VirtualMachine, VmCircuitConfig, VmState,
        },
        utils::test_system_config,
    };
    use openvm_instructions::{
        exe::VmExe,
        program::Program,
        riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
    };
    use openvm_native_compiler::conversion::AS;
    use openvm_stark_backend::{p3_field::PrimeField32, prover::hal::DeviceDataTransporter};
    use openvm_stark_sdk::{
        config::baby_bear_poseidon2::default_engine, openvm_stark_backend::engine::StarkEngine,
        p3_baby_bear::BabyBear,
    };
    use rand::{distributions::Standard, prelude::Distribution, rngs::StdRng, Rng};

    use crate::{extension::NativeConfig, Native};

    // If immediate, returns (value, AS::Immediate). Otherwise, writes to native memory and returns
    // (ptr, AS::Native). If is_imm is None, randomizes it.
    pub fn write_native_or_imm<F: PrimeField32>(
        tester: &mut VmChipTestBuilder<F>,
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
        tester: &mut VmChipTestBuilder<F>,
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

    pub fn execute_program_with_system_config(
        program: Program<BabyBear>,
        input_stream: impl Into<Streams<BabyBear>>,
        system_config: SystemConfig,
    ) -> eyre::Result<VmState<BabyBear>> {
        let config = NativeConfig::new(system_config, Native);
        let input = input_stream.into();

        let engine = default_engine();
        let pk = config.keygen(engine.config())?;
        let d_pk = engine.device().transport_pk_to_device(&pk);
        let vm = VirtualMachine::new(engine, config, d_pk)?;
        let ctx = vm.build_metered_ctx();
        let executor_idx_to_air_idx = vm.executor_idx_to_air_idx();
        let mut segments = vm.executor().execute_metered(
            program.clone(),
            input.clone(),
            &executor_idx_to_air_idx,
            ctx,
        )?;
        assert_eq!(segments.len(), 1, "test only supports one segment");
        let Segment {
            instret_start,
            num_insns,
            trace_heights,
        } = segments.pop().unwrap();
        assert_eq!(instret_start, 0);
        let exe = VmExe::new(program);
        let state = vm.executor().create_initial_state(&exe, input);
        let (_, _, state) = vm.execute_preflight(exe, state, num_insns, &trace_heights)?;
        Ok(state)
    }

    pub fn execute_program(
        program: Program<BabyBear>,
        input_stream: impl Into<Streams<BabyBear>>,
    ) -> VmState<BabyBear> {
        let system_config = test_native_config()
            .system
            .with_public_values(4)
            .with_max_segment_len((1 << 25) - 100);
        // we set max segment len large so it doesn't segment
        execute_program_with_system_config(program, input_stream, system_config).unwrap()
    }

    pub fn test_native_config() -> NativeConfig {
        let mut system = test_system_config();
        system.memory_config.addr_space_sizes[RV32_REGISTER_AS as usize] = 0;
        system.memory_config.addr_space_sizes[RV32_MEMORY_AS as usize] = 0;
        NativeConfig {
            system,
            native: Default::default(),
        }
    }

    pub fn test_native_continuations_config() -> NativeConfig {
        NativeConfig {
            system: test_system_config().with_continuations(),
            native: Default::default(),
        }
    }
}
