use std::sync::Arc;

use afs_primitives::xor_bits::XorBitsChip;
use criterion::{criterion_group, criterion_main, Criterion};
use p3_baby_bear::BabyBear;
use pprof::criterion::{Output, PProfProfiler};
use rand::Rng;

use afs_test_utils::utils::create_seeded_rng;

type Val = BabyBear;

pub fn xor_bits_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("trace gen");
    group.sample_size(100);

    let mut rng = create_seeded_rng();

    let bus_index = 0;

    const BITS: usize = 30;
    const MAX: u32 = 1 << BITS;

    const LOG_XOR_REQUESTS: usize = 20;
    const XOR_REQUESTS: usize = 1 << LOG_XOR_REQUESTS;

    let xor_chip = Arc::new(XorBitsChip::<BITS>::new(bus_index, vec![]));

    for _ in 0..XOR_REQUESTS {
        xor_chip.request(rng.gen::<u32>() % MAX, rng.gen::<u32>() % MAX);
    }

    group.bench_function("xor_bits", |b| b.iter(|| xor_chip.generate_trace::<Val>()));
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(10, Output::Flamegraph(None)));
    targets = xor_bits_benchmark
}

criterion_main!(benches);
