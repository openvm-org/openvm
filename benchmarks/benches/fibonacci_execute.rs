use ax_stark_sdk::p3_baby_bear::BabyBear;
use axvm_benchmarks::utils::build_bench_program;
use axvm_circuit::arch::{instructions::exe::AxVmExe, VmExecutor};
use axvm_rv32im_circuit::Rv32ImConfig;
use axvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use axvm_transpiler::{transpiler::Transpiler, FromElf};
use criterion::{criterion_group, criterion_main, Criterion};
use pprof::criterion::{Output, PProfProfiler};

fn benchmark_function(c: &mut Criterion) {
    let elf = build_bench_program("fibonacci").unwrap();
    let exe = AxVmExe::from_elf(
        elf,
        Transpiler::<BabyBear>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension),
    );

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
