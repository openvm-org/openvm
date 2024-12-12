use criterion::{criterion_group, criterion_main, Criterion};
use openvm_benchmarks::utils::build_bench_program;
use openvm_circuit::arch::{instructions::exe::VmExe, VmExecutor};
use openvm_rv32im_circuit::Rv32ImConfig;
use openvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use openvm_transpiler::{transpiler::Transpiler, FromElf};
use pprof::criterion::{Output, PProfProfiler};

fn benchmark_function(c: &mut Criterion) {
    let elf = build_bench_program("fibonacci").unwrap();
    let exe = VmExe::from_elf(
        elf,
        Transpiler::<BabyBear>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension),
    )
    .unwrap();

    let mut group = c.benchmark_group("fibonacci");
    let config = Rv32ImConfig::default();
    let executor = VmExecutor::<BabyBear, Rv32ImConfig>::new(config);

    group.bench_function("execute", |b| {
        b.iter(|| {
            executor.execute(exe.clone(), vec![]).unwrap();
        })
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = benchmark_function
}
criterion_main!(benches);
