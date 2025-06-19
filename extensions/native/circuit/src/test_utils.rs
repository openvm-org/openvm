use openvm_circuit::{
    arch::Streams,
    utils::{test_system_config, test_system_config_with_continuations},
};
use openvm_instructions::program::Program;
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use super::execute_program_with_system_config;
use crate::extension::NativeConfig;

pub fn test_execute_program(
    program: Program<BabyBear>,
    input_stream: impl Into<Streams<BabyBear>>,
) {
    let system_config = test_system_config()
        .with_public_values(4)
        .with_max_segment_len((1 << 25) - 100);
    execute_program_with_system_config(program, input_stream, system_config);
}

pub fn test_native_config() -> NativeConfig {
    NativeConfig {
        system: test_system_config(),
        native: Default::default(),
    }
}

pub fn test_native_continuations_config() -> NativeConfig {
    NativeConfig {
        system: test_system_config_with_continuations(),
        native: Default::default(),
    }
}
