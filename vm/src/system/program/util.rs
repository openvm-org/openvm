use axvm_instructions::program::Program;
use p3_baby_bear::BabyBear;

use crate::arch::{ExecutorName, VmConfig, VmExecutor};

pub fn execute_program_with_config(
    config: VmConfig,
    program: Program<BabyBear>,
    input_stream: Vec<Vec<BabyBear>>,
) {
    let executor = VmExecutor::<BabyBear>::new(config);
    executor.execute(program, input_stream).unwrap();
}

pub fn execute_program(program: Program<BabyBear>, input_stream: Vec<Vec<BabyBear>>) {
    let executor = VmExecutor::<BabyBear>::new(
        VmConfig {
            num_public_values: 4,
            max_segment_len: (1 << 25) - 100,
            ..Default::default()
        }
        .add_executor(ExecutorName::Phantom)
        .add_executor(ExecutorName::LoadStore)
        .add_executor(ExecutorName::BranchEqual)
        .add_executor(ExecutorName::Jal)
        .add_executor(ExecutorName::FieldArithmetic)
        .add_executor(ExecutorName::FieldExtension)
        .add_executor(ExecutorName::Poseidon2)
        .add_executor(ExecutorName::FriFold)
        .add_executor(ExecutorName::BaseAlu256Rv32)
        .add_executor(ExecutorName::LessThan256Rv32)
        .add_executor(ExecutorName::Multiplication256Rv32)
        .add_executor(ExecutorName::Shift256Rv32)
        .add_canonical_modulus(),
    );
    executor.execute(program, input_stream).unwrap();
}
