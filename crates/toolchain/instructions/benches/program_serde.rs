use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use openvm_instructions::{
    instruction::{Instruction, InstructionOperand},
    program::Program,
    VmOpcode,
};
use rand::prelude::*;

fn random_instruction(rng: &mut impl Rng) -> Instruction {
    let opcode = VmOpcode::from_usize(rng.random::<u16>() as usize);
    let mut operand = || rng.random_range(InstructionOperand::MIN..=InstructionOperand::MAX);
    Instruction::new(
        opcode,
        InstructionOperand::from_i32(operand()),
        InstructionOperand::from_i32(operand()),
        InstructionOperand::from_i32(operand()),
        InstructionOperand::from_i32(operand()),
        InstructionOperand::from_i32(operand()),
        InstructionOperand::from_i32(operand()),
        InstructionOperand::from_i32(operand()),
    )
}

fn program_serde_bench(c: &mut Criterion) {
    let mut rng = StdRng::from_seed([42; 32]);
    let instructions: Vec<_> = (0..100_000).map(|_| random_instruction(&mut rng)).collect();
    let program: Program = Program::from_instructions(&instructions);
    c.bench_function("bitcode serialize Program with 100000 instructions", |b| {
        b.iter(|| bitcode::serialize(black_box(&program)))
    });
    let bytes = bitcode::serialize(&program).unwrap();
    println!("Result length in bytes: {}", bytes.len());
    c.bench_function(
        "bitcode deserialize Program with 100000 instructions",
        |b| b.iter(|| bitcode::deserialize::<'_, Program>(black_box(&bytes))),
    );
}

criterion_group!(benches, program_serde_bench);
criterion_main!(benches);
