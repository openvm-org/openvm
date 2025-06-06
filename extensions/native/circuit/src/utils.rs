use openvm_circuit::arch::{
    execution_mode::metered::get_widths_and_interactions_from_vkey, Streams, SystemConfig,
    VirtualMachine,
};
use openvm_instructions::program::Program;
use openvm_stark_sdk::{config::baby_bear_poseidon2::default_engine, p3_baby_bear::BabyBear};

use crate::{Native, NativeConfig};

pub fn execute_program(program: Program<BabyBear>, input_stream: impl Into<Streams<BabyBear>>) {
    let system_config = SystemConfig::default()
        .with_public_values(4)
        .with_max_segment_len((1 << 25) - 100);
    let config = NativeConfig::new(system_config, Native);

    let input = input_stream.into();

    let vm = VirtualMachine::new(default_engine(), config);
    let pk = vm.keygen();
    let (widths, interactions) = get_widths_and_interactions_from_vkey(pk.get_vk());
    let segments = vm
        .executor
        .execute_metered(program.clone(), input.clone(), widths, interactions)
        .unwrap();
    vm.execute_with_segments(program, input, &segments).unwrap();
}

pub(crate) const fn const_max(a: usize, b: usize) -> usize {
    [a, b][(a < b) as usize]
}
