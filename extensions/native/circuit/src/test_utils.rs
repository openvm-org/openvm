use openvm_circuit::{arch::Streams, utils::test_system_config};
use openvm_instructions::program::Program;
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use super::execute_program_with_system_config;

pub fn test_execute_program(
    program: Program<BabyBear>,
    input_stream: impl Into<Streams<BabyBear>>,
) {
    let system_config = test_system_config()
        .with_public_values(4)
        .with_max_segment_len((1 << 25) - 100);
    execute_program_with_system_config(program, input_stream, system_config);
}
