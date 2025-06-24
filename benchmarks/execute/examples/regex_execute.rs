// use openvm_circuit::arch::{instructions::exe::VmExe, VmExecutor};
// use openvm_keccak256_circuit::Keccak256Rv32Config;
// use openvm_keccak256_transpiler::Keccak256TranspilerExtension;
// use openvm_rv32im_transpiler::{
//     Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
// };
// use openvm_sdk::StdIn;
// use openvm_stark_sdk::p3_baby_bear::BabyBear;
// use openvm_transpiler::{
//     elf::Elf, openvm_platform::memory::MEM_SIZE, transpiler::Transpiler, FromElf,
// };

fn main() {
    // let elf = Elf::decode(include_bytes!("regex-elf"), MEM_SIZE as u32).unwrap();
    // let exe = VmExe::from_elf(
    //     elf,
    //     Transpiler::<BabyBear>::default()
    //         .with_extension(Rv32ITranspilerExtension)
    //         .with_extension(Rv32MTranspilerExtension)
    //         .with_extension(Rv32IoTranspilerExtension)
    //         .with_extension(Keccak256TranspilerExtension),
    // )
    // .unwrap();

    // let config = Keccak256Rv32Config::default();
    // let executor = VmExecutor::<BabyBear, Keccak256Rv32Config>::new(config);

    // let data = include_str!("../../guest/regex/regex_email.txt");

    // let timer = std::time::Instant::now();
    // executor
    //     .execute(exe.clone(), StdIn::from_bytes(data.as_bytes()))
    //     .unwrap();
    // println!("execute_time: {:?}", timer.elapsed());
}
