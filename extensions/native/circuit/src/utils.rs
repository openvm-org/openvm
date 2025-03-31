use openvm_circuit::arch::{Streams, SystemConfig, VmExecutor};
use openvm_instructions::program::Program;
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use crate::{Native, NativeConfig};

pub fn execute_program(program: Program<BabyBear>, input_stream: impl Into<Streams<BabyBear>>) {
    let system_config = SystemConfig::default()
        .with_public_values(4)
        .with_max_segment_len((1 << 25) - 100);
    let config = NativeConfig::new(system_config, Native);
    let executor = VmExecutor::<BabyBear, NativeConfig>::new(config);

    executor.execute(program, input_stream).unwrap();
}

pub(crate) const fn const_max(a: usize, b: usize) -> usize {
    [a, b][(a < b) as usize]
}
