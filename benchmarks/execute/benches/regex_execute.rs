// TODO(ayush): add this back
// use criterion::{black_box, criterion_group, criterion_main, Criterion};
// use openvm_benchmarks_utils::{build_elf, get_programs_dir};
// use openvm_circuit::arch::{instructions::exe::VmExe, VmExecutor};
// use openvm_keccak256_circuit::Keccak256Rv32Config;
// use openvm_keccak256_transpiler::Keccak256TranspilerExtension;
// use openvm_rv32im_transpiler::{
//     Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
// };
// use openvm_sdk::StdIn;
// use openvm_stark_sdk::p3_baby_bear::BabyBear;
// use openvm_transpiler::{transpiler::Transpiler, FromElf};

// fn benchmark_function(c: &mut Criterion) {
//     let program_dir = get_programs_dir().join("regex");
//     let elf = build_elf(&program_dir, "release").unwrap();

//     let exe = VmExe::from_elf(
//         elf,
//         Transpiler::<BabyBear>::default()
//             .with_extension(Rv32ITranspilerExtension)
//             .with_extension(Rv32MTranspilerExtension)
//             .with_extension(Rv32IoTranspilerExtension)
//             .with_extension(Keccak256TranspilerExtension),
//     )
//     .unwrap();

//     let mut group = c.benchmark_group("regex");
//     group.sample_size(10);
//     let config = Keccak256Rv32Config::default();
//     let executor = VmExecutor::<BabyBear, Keccak256Rv32Config>::new(config);

//     let data = include_str!("../../guest/regex/regex_email.txt");

//     let fe_bytes = data.to_owned().into_bytes();
//     group.bench_function("execute", |b| {
//         b.iter(|| {
//             let input = black_box(Stdin::from_bytes(&fe_bytes));
//             executor.execute(exe.clone(), input).unwrap();
//         })
//     });

//     group.finish();
// }

// criterion_group!(benches, benchmark_function);
// criterion_main!(benches);
