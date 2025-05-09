use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use openvm_instructions::{instruction::Instruction, program::Program, VmOpcode};
use p3_baby_bear::BabyBear;
use rand::prelude::*;

type F = BabyBear;

fn random_instruction(rng: &mut impl Rng) -> Instruction<F> {
    Instruction::new(
        VmOpcode::from_usize(rng.gen()),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
    )
}

fn program_serde_bench(c: &mut Criterion) {
    let mut rng = StdRng::from_seed([42; 32]);
    let instructions: Vec<_> = (0..100_000).map(|_| random_instruction(&mut rng)).collect();
    let program: Program<F> = Program::from_instructions(&instructions);
    c.bench_function("bitcode serialize Program with 100000 instructions", |b| {
        b.iter(|| bitcode::serialize(black_box(&program)))
    });
    let bytes = bitcode::serialize(&program).unwrap();
    println!("Result length in bytes: {}", bytes.len());
    c.bench_function(
        "bitcode deserialize Program with 100000 instructions",
        |b| b.iter(|| bitcode::deserialize::<'_, Program<F>>(black_box(&bytes))),
    );
}

criterion_group!(benches, program_serde_bench);
criterion_main!(benches);
