// use criterion::{criterion_group, criterion_main, Criterion};
// use openvm_benchmarks_utils::{build_elf, get_programs_dir};
// use openvm_circuit::arch::{instructions::exe::VmExe, VmExecutor};
// use openvm_rv32im_circuit::Rv32ImConfig;
// use openvm_rv32im_transpiler::{
//     Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
// };
// // TODO(ayush): add this back
// // use openvm_sdk::StdIn;
// use openvm_stark_sdk::p3_baby_bear::BabyBear;
// use openvm_transpiler::{transpiler::Transpiler, FromElf};

// fn benchmark_function(c: &mut Criterion) {
//     let program_dir = get_programs_dir().join("fibonacci");
//     let elf = build_elf(&program_dir, "release").unwrap();

//     let exe = VmExe::from_elf(
//         elf,
//         Transpiler::<BabyBear>::default()
//             .with_extension(Rv32ITranspilerExtension)
//             .with_extension(Rv32MTranspilerExtension)
//             .with_extension(Rv32IoTranspilerExtension),
//     )
//     .unwrap();

//     let mut group = c.benchmark_group("fibonacci");
//     let config = Rv32ImConfig::default();
//     let executor = VmExecutor::<BabyBear, Rv32ImConfig>::new(config);

//     group.bench_function("execute", |b| {
//         b.iter(|| {
//             // TODO(ayush): add this back
//             // let n = 100_000u64;
//             // let mut stdin = StdIn::default();
//             // stdin.write(&n);
//             executor.execute(exe.clone(), vec![]).unwrap();
//         })
//     });

//     group.finish();
// }

// criterion_group!(benches, benchmark_function);
// criterion_main!(benches);
