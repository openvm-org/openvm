use openvm_circuit::arch::{Streams, SystemConfig, VirtualMachine};
use openvm_instructions::program::Program;
use openvm_stark_sdk::{config::baby_bear_poseidon2::default_engine, p3_baby_bear::BabyBear};

use crate::{Native, NativeConfig};

pub(crate) const CASTF_MAX_BITS: usize = 30;

pub fn execute_program_with_system_config(
    program: Program<BabyBear>,
    input_stream: impl Into<Streams<BabyBear>>,
    system_config: SystemConfig,
) {
    let config = NativeConfig::new(system_config, Native);
    let input = input_stream.into();

    let vm = VirtualMachine::new(default_engine(), config);
    let pk = vm.keygen();
    let vk = pk.get_vk();
    let segments = vm
        .executor
        .execute_metered(program.clone(), input.clone(), &vk.num_interactions())
        .unwrap();
    vm.execute(program, input, &segments).unwrap();
}

pub fn execute_program(program: Program<BabyBear>, input_stream: impl Into<Streams<BabyBear>>) {
    let system_config = SystemConfig::default()
        .with_public_values(4)
        .with_max_segment_len((1 << 25) - 100);
    execute_program_with_system_config(program, input_stream, system_config);
}

pub(crate) const fn const_max(a: usize, b: usize) -> usize {
    [a, b][(a < b) as usize]
}

/// Testing framework
#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils {
    use std::array;

    use openvm_circuit::{
        arch::{
            testing::{memory::gen_pointer, VmChipTestBuilder},
            Streams,
        },
        utils::test_system_config,
    };
    use openvm_instructions::{
        program::Program,
        riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
    };
    use openvm_native_compiler::conversion::AS;
    use openvm_stark_backend::p3_field::PrimeField32;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use rand::{distributions::Standard, prelude::Distribution, rngs::StdRng, Rng};

    use super::execute_program_with_system_config;
    use crate::{extension::NativeConfig, Rv32WithKernelsConfig};

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

    pub fn test_execute_program(
        program: Program<BabyBear>,
        input_stream: impl Into<Streams<BabyBear>>,
    ) {
        let system_config = test_native_config()
            .system
            .with_public_values(4)
            .with_max_segment_len((1 << 25) - 100);
        execute_program_with_system_config(program, input_stream, system_config);
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

    pub fn test_rv32_with_kernels_config() -> Rv32WithKernelsConfig {
        Rv32WithKernelsConfig {
            system: test_system_config().with_continuations(),
            ..Default::default()
        }
    }
}
