use axvm_instructions::program::Program;
use p3_baby_bear::BabyBear;

use crate::arch::{VmConfig, VmExecutor};

pub fn execute_program_with_config(
    config: VmConfig,
    program: Program<BabyBear>,
    input_stream: Vec<Vec<BabyBear>>,
) {
    let executor = VmExecutor::<BabyBear>::new(config);
    executor.execute(program, input_stream).unwrap();
}
